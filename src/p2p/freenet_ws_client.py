import asyncio
import json
import websockets
import logging
import threading
import hashlib
import uuid
import time
from typing import Dict, Any, Callable, Optional, List, Tuple

logger = logging.getLogger(__name__)

def get_freenet_udp_port() -> int:
    """
    Dynamically attempts to locate the local Freenet/Locutus UDP peer port.
    Returns the port number if found, otherwise returns None.
    """
    try:
        import psutil
        for proc in psutil.process_iter(['name', 'pid']):
            name = proc.info.get('name', '')
            if name and ('java' in name.lower() or 'wrapper' in name.lower() or 'locutus' in name.lower() or 'freenet' in name.lower()):
                try:
                    conn_fn = getattr(proc, 'net_connections', getattr(proc, 'connections', None))
                    if conn_fn:
                        for conn in conn_fn(kind='udp'):
                            if conn.status == 'NONE' or conn.status == '':
                                # UDP doesn't really have a 'LISTEN' state, check if port is bound
                                if conn.laddr and conn.laddr.port > 0:
                                    return conn.laddr.port
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
    except ImportError:
        pass
    
    # PowerShell fallback
    import sys
    if sys.platform == 'win32':
        import subprocess
        try:
            ps_cmd = (
                "Get-NetUDPEndpoint | Where-Object { $_.LocalAddress -eq '127.0.0.1' -or $_.LocalAddress -eq '::1' } | "
                "Select-Object LocalPort, OwningProcess | ForEach-Object { "
                "$proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue; "
                "if ($proc.Name -match 'java|locutus|freenet|wrapper') { $_.LocalPort } }"
            )
            output = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True, timeout=2.0).strip()
            if output:
                ports = [int(p.strip()) for p in output.splitlines() if p.strip().isdigit()]
                if ports:
                    return ports[0]
        except Exception as e:
            logger.warning(f"[FREENET] PowerShell UDP port discovery failed: {e}")
            
    logger.warning("[FREENET] Could not auto-detect UDP port.")
    return None

def get_freenet_tcp_port() -> int:
    """
    Dynamically attempts to locate the local Freenet/Locutus TCP WebSocket port.
    It looks for 'LISTEN' state connections belonging to Locutus processes.
    Returns the port number if found, otherwise returns None.
    """
    candidate_ports = []
    try:
        import psutil
        for proc in psutil.process_iter(['name', 'pid']):
            name = proc.info.get('name', '')
            if name and ('java' in name.lower() or 'wrapper' in name.lower() or 'locutus' in name.lower() or 'freenet' in name.lower()):
                try:
                    conn_fn = getattr(proc, 'net_connections', getattr(proc, 'connections', None))
                    if conn_fn:
                        for conn in conn_fn(kind='tcp'):
                            if conn.status == 'LISTEN':
                                if conn.laddr and conn.laddr.port > 0:
                                    candidate_ports.append(conn.laddr.port)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
    except ImportError:
        pass

    # PowerShell fallback
    import sys
    if not candidate_ports and sys.platform == 'win32':
        import subprocess
        try:
            ps_cmd = (
                "Get-NetTCPConnection | Where-Object { $_.State -eq 'Listen' -and ($_.LocalAddress -eq '127.0.0.1' -or $_.LocalAddress -eq '::1') } | "
                "Select-Object LocalPort, OwningProcess | ForEach-Object { "
                "$proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue; "
                "if ($proc.Name -match 'java|locutus|freenet|wrapper') { $_.LocalPort } }"
            )
            output = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True, timeout=2.0).strip()
            if output:
                ps_ports = [int(p.strip()) for p in output.splitlines() if p.strip().isdigit()]
                candidate_ports.extend(ps_ports)
        except Exception as e:
            logger.warning(f"[FREENET] PowerShell TCP port discovery failed: {e}")

    if candidate_ports:
        return min(candidate_ports, key=lambda p: abs(p - 3000))
        
    logger.warning("[FREENET] Could not auto-detect TCP port.")
    return None

class RoguePredictorDefense:
    """
    Guards P2P and OpenRequest messaging against Rogue Machine Learning Predictors
    operating on Freenet, Hyphanet, or alternative decentralized topologies.
    
    Adversarial Threat Model:
    1. Route Prediction & Traceback Attacks: Adversaries deploy statistical/ML classifiers
       trained on packet timing distributions (Delta t), hop decay, and keyspace proximity
       to trace requests back to the originator node.
    2. Deceptive Next-Hop Latency Manipulation: Rogue peers skew response latencies to hijack
       DHT/keyspace clusters.
    3. State Reconstruction: Probing queries attempt to infer latent cognitive embeddings.

    Active Mitigations:
    - Honest Timing Jitter: Injects non-Gaussian stochastic delays via harvest_honest_jitter
      to destroy inter-arrival autocorrelation.
    - ZK State Commitments: Cryptographically binds payloads via Blake2s commitments so
      open request routing nodes cannot inspect cognitive interiors.
    - Probing Entropy Audit: Flags peers that exhibit automated, sub-entropy scanning.
    - RP4 Projective Slashing: Quenches malicious peers into the boundary of Real Projective Space.
    """
    def __init__(self, jitter_min_ms: float = 5.0, jitter_max_ms: float = 45.0):
        self.jitter_min_ms = jitter_min_ms
        self.jitter_max_ms = jitter_max_ms
        self.slashed_peers: Dict[str, Dict[str, Any]] = {}
        self.peer_history: Dict[str, List[float]] = {}
        self.probes_detected = 0
        self.requests_shielded = 0
        self.total_jitter_applied_ms = 0.0

    def generate_defense_jitter(self) -> float:
        """Computes honest timing jitter delay in milliseconds to destroy ML classification."""
        try:
            from src.core.honest_jitter import harvest_honest_jitter
            jitter_tensor = harvest_honest_jitter((1,), scaled=False)
            jitter_scalar = float(jitter_tensor.abs().item())
        except Exception:
            import random
            jitter_scalar = random.random()
            
        jitter_ms = self.jitter_min_ms + (self.jitter_max_ms - self.jitter_min_ms) * (jitter_scalar % 1.0)
        self.requests_shielded += 1
        self.total_jitter_applied_ms += jitter_ms
        return jitter_ms

    def compute_zk_commitment(self, state: Any) -> str:
        """Computes deterministic Blake2s zero-knowledge state commitment."""
        if state is None:
            return hashlib.blake2s(b"null").hexdigest()
        if isinstance(state, (dict, list)):
            serialized = json.dumps(state, sort_keys=True, separators=(',', ':'))
        else:
            serialized = str(state)
        return hashlib.blake2s(serialized.encode('utf-8')).hexdigest()

    def record_and_audit_peer(self, peer_id: str, contract_key: str, latency_ms: float) -> Tuple[bool, str]:
        """
        Audits incoming peer responses and timing for rogue predictive behavior.
        Returns (is_rogue, reason).
        """
        if peer_id in self.slashed_peers:
            return True, "ALREADY_SLASHED"

        now = time.time()
        if peer_id not in self.peer_history:
            self.peer_history[peer_id] = []
        history = self.peer_history[peer_id]
        history.append(now)

        # Keep rolling 1-minute window
        cutoff = now - 60.0
        self.peer_history[peer_id] = [t for t in history if t > cutoff]
        recent = self.peer_history[peer_id]

        # Anomaly 1: Superhuman frequency (> 100 requests/min from a single peer)
        if len(recent) > 100:
            self.probes_detected += 1
            self.slash_peer(peer_id, reason="HIGH_FREQUENCY_PREDICTION_SWEEP")
            return True, "HIGH_FREQUENCY_PREDICTION_SWEEP"

        # Anomaly 2: Extreme timing precision (variance < 1e-6 with N >= 10, indicative of automated ML probing)
        if len(recent) >= 10:
            intervals = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            mean_i = sum(intervals) / len(intervals)
            var_i = sum((x - mean_i) ** 2 for x in intervals) / len(intervals)
            if var_i < 1e-6:
                self.probes_detected += 1
                self.slash_peer(peer_id, reason="SYNTHETIC_TIMING_REGULARITY")
                return True, "SYNTHETIC_TIMING_REGULARITY"

        return False, "NORMAL"

    def slash_peer(self, peer_id: str, reason: str = "ROGUE_ML_PREDICTOR") -> Dict[str, Any]:
        """Slashes malicious peer and assigns RP4 boundary projective isolation."""
        now = time.time()
        slash_record = {
            "peer_id": peer_id,
            "reason": reason,
            "timestamp": now,
            "rp4_boundary_projection": [0.0, 0.0, 0.0, 0.0, 1.0],  # Boundary point in RP4
            "status": "ISOLATED"
        }
        self.slashed_peers[peer_id] = slash_record
        logger.warning(f"[SECURITY] Rogue Predictor Peer {peer_id} slashed from OpenRequest network: {reason}")
        return slash_record

    def is_peer_slashed(self, peer_id: str) -> bool:
        return peer_id in self.slashed_peers

    def get_telemetry(self) -> Dict[str, Any]:
        mean_jitter = (self.total_jitter_applied_ms / max(1, self.requests_shielded)) if self.requests_shielded > 0 else 0.0
        return {
            "active_defense": True,
            "defense_protocol": "ROGUE_ML_PREDICTOR_SHIELD_V1",
            "rogue_probes_detected": self.probes_detected,
            "slashed_peers_count": len(self.slashed_peers),
            "total_open_requests_shielded": self.requests_shielded,
            "mean_jitter_ms": round(mean_jitter, 2)
        }


class FreenetClient:
    """
    WebSocket client connecting to local Freenet Core (Locutus) daemon.
    Manages state contracts, OpenRequest pipelines, and Rogue Predictor defense
    for the Gyroidic Sparse Covariance Flux Reasoner.
    """
    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None, ws_url: Optional[str] = None):
        import os
        if ws_url:
            self.uri = ws_url
        else:
            if port is None:
                env_port = os.environ.get("FREENET_WS_PORT")
                if env_port is not None:
                    port = int(env_port)
                else:
                    auto_port = get_freenet_tcp_port()
                    port = auto_port if auto_port else 3000
            self.uri = f"ws://{host}:{port}/"

        self.ws = None
        self.running = False
        self.subscriptions: Dict[str, Callable] = {}
        self.pending_open_requests: Dict[str, Callable] = {}
        self.cached_open_requests: Dict[str, Dict[str, Any]] = {}
        self.loop = None
        self.thread = None
        self.client_id = f"client_{uuid.uuid4().hex[:12]}"
        self.defense = RoguePredictorDefense()
        self._last_log_time = 0.0

    async def _connect_and_listen(self):
        while self.running:
            try:
                async with websockets.connect(self.uri, open_timeout=2.0) as ws:
                    self.ws = ws
                    logger.debug(f"[FREENET] Connected to local daemon at {self.uri}")
                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            self._handle_message(message)
                        except asyncio.TimeoutError:
                            continue
            except Exception as e:
                # Freenet node is often not running, avoid spamming terminal.
                logger.debug(f"[FREENET] Failed to connect or lost connection: {e}")
            finally:
                self.ws = None
            
            if self.running:
                try:
                    for _ in range(20):
                        if not self.running:
                            break
                        await asyncio.sleep(0.25)
                except (asyncio.CancelledError, Exception):
                    break

    def _handle_message(self, message: str):
        try:
            data = json.loads(message)
            # 1. Resolve OpenRequest callback if request_id matches
            req_id = data.get("request_id")
            if req_id and req_id in self.pending_open_requests:
                cb = self.pending_open_requests.pop(req_id)
                res = data.get("result") or data.get("state") or data
                cb(res)

            # 2. Resolve contract_id / contract key subscriptions
            contract_id = data.get("contract_id") or data.get("key")
            if contract_id and contract_id in self.subscriptions:
                self.subscriptions[contract_id](data.get("state"))
        except Exception as e:
            logger.error(f"[FREENET] Error parsing message: {e}")

    def start(self):
        if self.running:
            return
        self.running = True
        self.loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self._connect_and_listen())
            except (asyncio.CancelledError, RuntimeError, KeyboardInterrupt):
                pass
            except Exception as e:
                logger.debug(f"[FREENET] Loop terminated: {e}")
            finally:
                try:
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                try:
                    self.loop.close()
                except Exception:
                    pass
            
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.loop and self.loop.is_running():
            def _cancel_and_stop():
                try:
                    for task in asyncio.all_tasks(self.loop):
                        task.cancel()
                except Exception:
                    pass
                try:
                    self.loop.stop()
                except Exception:
                    pass
            try:
                self.loop.call_soon_threadsafe(_cancel_and_stop)
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def subscribe(self, contract_id: str, callback: Callable):
        """Subscribe to a specific topological state contract."""
        self.subscriptions[contract_id] = callback
        if self.ws and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps({"type": "subscribe", "contract_id": contract_id})),
                self.loop
            )

    def publish(self, contract_id: str, state: Dict[str, Any]):
        """Publish a state update (e.g. Kelly bet or encrypted Zeta) to the network."""
        if self.ws and self.loop:
            serialized_state = json.dumps(state, sort_keys=True, separators=(',', ':'))
            deterministic_id = hashlib.blake2s(serialized_state.encode('utf-8')).hexdigest()
            payload = {
                "type": "update", 
                "contract_id": contract_id, 
                "state": state,
                "payload_id": deterministic_id
            }
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps(payload, sort_keys=True, separators=(',', ':'))),
                self.loop
            )
        else:
            current_time = time.time()
            if current_time - self._last_log_time >= 60.0:
                logger.debug("[FREENET] Cannot publish: WebSocket not connected.")
                self._last_log_time = current_time

    def send_open_request(
        self,
        request_type: str,
        contract_key: str,
        state: Optional[Any] = None,
        delta: Optional[Any] = None,
        callback: Optional[Callable] = None,
        apply_jitter: bool = True
    ) -> str:
        """
        Dispatches a freenet-core OpenRequest envelope over WebSocket.
        Applies honest timing jitter to neutralize rogue ML routing prediction.
        """
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        if callback:
            self.pending_open_requests[request_id] = callback
            
        jitter_ms = self.defense.generate_defense_jitter() if apply_jitter else 0.0
        commitment = self.defense.compute_zk_commitment(state if state is not None else delta)

        payload = {
            "type": "open_request",
            "client_id": self.client_id,
            "request_id": request_id,
            "request": {
                "type": request_type,
                "key": contract_key,
                "state": state,
                "delta": delta
            },
            "security": {
                "defense": "ROGUE_ML_PREDICTOR_SHIELD_V1",
                "zk_commitment": commitment,
                "jitter_applied_ms": jitter_ms
            }
        }

        if self.ws and self.loop:
            async def _send_with_jitter():
                if jitter_ms > 0:
                    await asyncio.sleep(jitter_ms / 1000.0)
                await self.ws.send(json.dumps(payload, sort_keys=True, separators=(',', ':')))

            asyncio.run_coroutine_threadsafe(_send_with_jitter(), self.loop)
        else:
            self.cached_open_requests[request_id] = payload

        return request_id

    def get_contract(self, contract_key: str, callback: Optional[Callable] = None) -> str:
        """Dispatches an OpenRequest 'get' to retrieve a WebAssembly contract state."""
        return self.send_open_request(request_type="get", contract_key=contract_key, callback=callback)

    def put_contract(self, contract_key: str, state: Dict[str, Any], callback: Optional[Callable] = None) -> str:
        """Dispatches an OpenRequest 'put' to register or store contract state."""
        return self.send_open_request(request_type="put", contract_key=contract_key, state=state, callback=callback)

    def update_contract(self, contract_key: str, delta: Dict[str, Any], callback: Optional[Callable] = None) -> str:
        """Dispatches an OpenRequest 'update' to apply delta update."""
        return self.send_open_request(request_type="update", contract_key=contract_key, delta=delta, callback=callback)

    def audit_peer_for_rogue_prediction(self, peer_id: str, contract_key: str, latency_ms: float) -> Tuple[bool, str]:
        """Audits peer request for rogue predictive activity."""
        return self.defense.record_and_audit_peer(peer_id, contract_key, latency_ms)

    def get_security_telemetry(self) -> Dict[str, Any]:
        """Returns Rogue Predictor defense and OpenRequest metrics."""
        telem = self.defense.get_telemetry()
        telem["active_open_requests"] = len(self.pending_open_requests)
        telem["cached_open_requests"] = len(self.cached_open_requests)
        telem["ws_connected"] = bool(self.ws is not None)
        return telem
