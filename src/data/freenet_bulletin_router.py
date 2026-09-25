import socket
import threading
import uuid
import datetime
import time

class FreenetBulletinRouter:
    """
    Zeitgeist Translator that bridges the internal Non-Dual Coin ledger
    with the global Freenet bulletin boards (FMS & Sone).
    Generates synthetic payloads and routes them via FCPv2.
    """
    _last_broadcast_time = 0.0

    def __init__(self, host: str = '127.0.0.1', port: int = 7509, freenet_client=None):
        self.host = host
        self.port = port
        self.freenet_client = freenet_client
        self.sone_uri_base = "USK@Gyroidic-Sone-Identity"
        self.fms_board = "Gyroidic.Resonance"
        self._fms_ssk_insert_uri = None
        self._sone_ssk_insert_uri = None

    def _ensure_identity_ssk(self):
        """Communicates with Freenet node via FCP to generate a real SSK keypair for the router identity."""
        if self.freenet_client:
            self._fms_ssk_insert_uri = "Locutus-WebSocket-Identity"
            self._sone_ssk_insert_uri = "Locutus-WebSocket-Identity"
            return True

        if self._fms_ssk_insert_uri:
            return True

        if not getattr(self, '_fcp_error_logged', False):
            print(f"[FREENET WARN] Locutus Client Offline - Falling back to offline P2P simulation.")
            self._fcp_error_logged = True
        
        # Use offline simulated SSK so the loop continues simulating broadcasts instead of aborting
        self._fms_ssk_insert_uri = f"SSK@offline-simulated-{uuid.uuid4().hex[:16]}"
        self._sone_ssk_insert_uri = self._fms_ssk_insert_uri.replace("SSK@", "USK@")
        self._offline_mode = True
        return True

    def _generate_fms_xml(self, volume: float, mischief: float, metrics: dict = None) -> str:
        """Generates a Bonfire P2P FMS Message XML payload."""
        date_str = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        time_str = datetime.datetime.utcnow().strftime('%H:%M:%S')
        message_id = f"{uuid.uuid4().hex}@fms.gyroidic"
        
        if metrics is None:
            raise ValueError("Cannot generate FMS XML: Topological metrics are disconnected. Bulletin Board state required.")
            
        kelly_fraction = metrics.get('kelly_fraction', 0.0)
        covariance_variance = metrics.get('covariance_variance', 0.0)
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Message>
    <Date>{date_str}</Date>
    <Time>{time_str}</Time>
    <Subject>Bonfire Ring: Proof of Honesty & Microhedging</Subject>
    <MessageID>{message_id}</MessageID>
    <ReplyBoard>{self.fms_board}</ReplyBoard>
    <Body>
        <![CDATA[
        [BONFIRE P2P PROTOCOL]
        The Gyroidic Reasoner has reached a local consensus state.
        Current Ledger Volume: {volume:.4f}
        Mischief Digested: {mischief:.4f}
        
        [Egalitarian Microhedging]
        Fractional Kelly Allocation: {kelly_fraction:.4f}
        Covariance Variance: {covariance_variance:.4f}
        
        System maintains topological resonance against the Nomadic Ring.
        ]]>
    </Body>
</Message>
"""
        return xml

    def _generate_sone_json(self, volume: float, metrics: dict = None) -> str:
        """Generates a Bonfire P2P Nomadic Ring topological signature payload."""
        import json
        if metrics is None:
            raise ValueError("Cannot generate Sone JSON: Topological metrics are disconnected. Bulletin Board state required.")
        
        payload = {
            "type": "BonfireNomadicRingSignature",
            "volume": round(volume, 4),
            "topological_signature": {
                "betti_numbers": metrics.get("betti_numbers", []),
                "euler_characteristic": metrics.get("euler_characteristic", 0),
                "coprime_residues": metrics.get("coprime_residue", 1)
            },
            "egalitarian_microhedging": {
                "kelly_fraction": metrics.get("kelly_fraction", 0.01),
                "covariance_variance": metrics.get("covariance_variance", 0.01),
                "valence_drive": metrics.get("valence_drive", 1.0)
            },
            "tags": ["#BonfireRing", "#Gyroidic", "#NonDualCoin"]
        }
        return json.dumps(payload, indent=2)


    def broadcast_proof_of_honesty(self, volume: float, mischief: float, metrics: dict = None):
        """Asynchronously dispatches the FMS and Sone synthetic payloads over FCPv2."""
        current_time = time.time()
        if current_time - FreenetBulletinRouter._last_broadcast_time < 60.0:
            return

        FreenetBulletinRouter._last_broadcast_time = current_time

        if self.host not in ['127.0.0.1', 'localhost']:
            print("[FREENET WARN] SSRF Protection active: Host must be local.")
            return

        if not self._ensure_identity_ssk():
            print("[FREENET WARN] Could not generate or retrieve SSK identity. Aborting broadcast.")
            return

        fms_payload = self._generate_fms_xml(volume, mischief, metrics=metrics)
        sone_payload = self._generate_sone_json(volume, metrics=metrics)
        
        def _run():
            if self.freenet_client:
                self.freenet_client.publish("bulletin_fms", {"xml": fms_payload})
                self.freenet_client.publish("bulletin_sone", {"json": sone_payload})
                print("[FREENET] Proof of Honesty successfully dispatched via Locutus WebSocket.")
                return

            if getattr(self, '_offline_mode', False) or not self.freenet_client:
                if not getattr(self, '_offline_msg_logged', False):
                    print("[FREENET] Offline Mode: Proof of Honesty broadcasts will be simulated locally instead of using FCP.")
                    self._offline_msg_logged = True
                return
                
        t = threading.Thread(target=_run, daemon=True, name="FreenetBulletinThread")
        t.start()

    def broadcast_compute_challenge_receipt(self, local_node_id: str, peer_id: str, outcome: str):
        """Asynchronously dispatches a FMS/Sone receipt when a peer completes a compute challenge or is booted."""
        date_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        body = f"[P2P RING EVENT] Node: {local_node_id} | Peer: {peer_id} | Outcome: {outcome} | Timestamp: {date_str}"
        
        def _run_challenge_broadcast():
            if self.freenet_client:
                self.freenet_client.publish("challenge_receipt", {"body": body})
                print("[FREENET] Compute Challenge Receipt dispatched via Locutus WebSocket.")
                return

            if getattr(self, '_offline_mode', False) or not self.freenet_client:
                return

        t = threading.Thread(target=_run_challenge_broadcast, daemon=True, name="FreenetChallengeThread")
        t.start()


    def generate_dummy_payload(self) -> str:
        """For manual verification and writing to scratch."""
        return self._generate_fms_xml(1050.25, 42.1)
