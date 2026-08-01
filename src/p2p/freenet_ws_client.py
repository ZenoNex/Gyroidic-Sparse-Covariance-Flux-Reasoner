import asyncio
import json
import websockets
import logging
import threading
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

class FreenetClient:
    """
    WebSocket client to connect to a local Freenet Core (Locutus) daemon.
    Manages state contracts for the Gyroidic Sparse Covariance Flux Reasoner.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 3000):
        self.uri = f"ws://{host}:{port}/"
        self.ws = None
        self.running = False
        self.subscriptions: Dict[str, Callable] = {}
        self.loop = None
        self.thread = None

    async def _connect_and_listen(self):
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
            logger.error(f"[FREENET] Failed to connect or lost connection: {e}")
        finally:
            self.ws = None

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
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps({"type": "update", "contract_id": contract_id, "state": state})),
                self.loop
            )
        else:
            logger.warning("[FREENET] Cannot publish: WebSocket not connected.")
