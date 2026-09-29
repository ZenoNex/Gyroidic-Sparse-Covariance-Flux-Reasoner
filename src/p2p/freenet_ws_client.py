import asyncio
import json
import websockets
import logging
import threading
import hashlib
from typing import Dict, Any, Callable

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
                    for conn in proc.connections(kind='udp'):
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
            output = subprocess.check_output(["powershell", "-Command", ps_cmd], text=True).strip()
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
                    for conn in proc.connections(kind='tcp'):
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
            output = subprocess.check_output(["powershell", "-Command", ps_cmd], text=True).strip()
            if output:
                ps_ports = [int(p.strip()) for p in output.splitlines() if p.strip().isdigit()]
                candidate_ports.extend(ps_ports)
        except Exception as e:
            logger.warning(f"[FREENET] PowerShell TCP port discovery failed: {e}")

    if candidate_ports:
        return min(candidate_ports, key=lambda p: abs(p - 3000))
        
    logger.warning("[FREENET] Could not auto-detect TCP port.")
    return None

class FreenetClient:
    """
    WebSocket client to connect to a local Freenet Core (Locutus) daemon.
    Manages state contracts for the Gyroidic Sparse Covariance Flux Reasoner.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = None):
        import os
        if port is None:
            # 1. Check environment variable
            env_port = os.environ.get("FREENET_WS_PORT")
            if env_port is not None:
                port = int(env_port)
            else:
                # 2. Dynamically auto-detect TCP port via psutil/OS polling
                auto_port = get_freenet_tcp_port()
                port = auto_port if auto_port else 3000
                
        self.uri = f"ws://{host}:{port}/"
        self.ws = None
        self.running = False
        self.subscriptions: Dict[str, Callable] = {}
        self.loop = None
        self.thread = None
        self._last_log_time = 0.0

    async def _connect_and_listen(self):
        while self.running:
            try:
                async with websockets.connect(self.uri) as ws:
                    self.ws = ws
                    logger.info(f"[FREENET] Connected to local daemon at {self.uri}")
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
                await asyncio.sleep(5.0)

    def _handle_message(self, message: str):
        try:
            data = json.loads(message)
            contract_id = data.get("contract_id")
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
            self.loop.run_until_complete(self._connect_and_listen())
            
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.loop:
            self.loop.stop()

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
            # ASD-STE100 Rules: Deterministic Blake2s Serialization
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
            import time
            current_time = time.time()
            if current_time - self._last_log_time >= 60.0:
                logger.warning("[FREENET] Cannot publish: WebSocket not connected.")
                self._last_log_time = current_time
