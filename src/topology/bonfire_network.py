"""
Nomadic Ring Protocol & Bonfire Peer-to-Peer Subsystem.

Role: Orchestrates peer-to-peer rings to share topological invariants, distribute
heavy computations to synthetic endpoints, and perform egalitarian Kelly consensus sharing
asynchronously using a background daemon thread to avoid blocking the main reasoner loop.
"""

import json
import os
import urllib.request
import urllib.error
import threading
import time
from typing import List, Dict, Any, Optional
import torch
from concurrent.futures import ThreadPoolExecutor

PEERS_FILE = os.path.join("data", "bonfire_peers.json")

class BonfireNetwork:
    def __init__(self, node_id: str = "node_alpha", local_url: str = "http://localhost:8000"):
        self.node_id = node_id
        self.local_url = local_url
        self.peers: List[str] = []
        
        # Thread-safe caches
        self.lock = threading.Lock()
        self.cached_signatures: Dict[str, str] = {}
        self.cached_consensus_kelly: float = 0.5
        self.cached_consensus_p_success: float = 0.5
        self.healthy_peers: List[str] = []
        
        self.local_signature: str = "init_sig"
        self.local_kelly: float = 0.5
        self.local_p_success: float = 0.5
        
        self._load_peers()
        
        # Async worker thread pool
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Start periodic background sync daemon
        self.running = True
        self.daemon_thread = threading.Thread(target=self._background_loop, daemon=True)
        self.daemon_thread.start()

    def _load_peers(self):
        """Loads registered peer endpoints from disk."""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(PEERS_FILE):
            # Seed with default P2P ports
            self.peers = ["http://localhost:8001", "http://localhost:8002"]
            self._save_peers()
        else:
            try:
                with open(PEERS_FILE, "r") as f:
                    data = json.load(f)
                    self.peers = data.get("peers", [])
            except Exception:
                self.peers = []

    def _save_peers(self):
        """Saves current peer list to disk."""
        try:
            with open(PEERS_FILE, "w") as f:
                json.dump({"peers": self.peers}, f, indent=4)
        except Exception as e:
            print(f"[BONFIRE] Failed to save peers: {e}")

    def add_peer(self, url: str):
        with self.lock:
            if url not in self.peers and url != self.local_url:
                self.peers.append(url)
                self._save_peers()
                print(f"[BONFIRE] Added nomadic ring peer: {url}")

    def remove_peer(self, url: str):
        with self.lock:
            if url in self.peers:
                self.peers.remove(url)
                self._save_peers()
                print(f"[BONFIRE] Removed peer: {url}")

    def _background_loop(self):
        """Background loop to periodically coordinate with peers without blocking the reasoner."""
        while self.running:
            time.sleep(10.0)  # Coordinate every 10 seconds
            
            with self.lock:
                peers_to_check = list(self.peers)
                local_sig = self.local_signature
                local_k = self.local_kelly
                local_p = self.local_p_success
                
            active_healthy = []
            sigs = {}
            kellies = [local_k]
            successes = [local_p]
            
            for peer in peers_to_check:
                # 1. Coordinate /api/bonfire/sync
                try:
                    url = f"{peer}/api/bonfire/sync"
                    payload = json.dumps({"node_id": self.node_id, "signature": local_sig}).encode('utf-8')
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={"Content-Type": "application/json", "User-Agent": "BonfireRingNode/1.9"}
                    )
                    with urllib.request.urlopen(req, timeout=1.0) as response:
                        res_data = json.loads(response.read().decode('utf-8'))
                        sig = res_data.get("signature")
                        if sig:
                            sigs[peer] = sig
                    active_healthy.append(peer)
                except Exception:
                    continue  # Peer unreachable or timed out
                    
                # 2. Coordinate /api/bonfire/kelly
                try:
                    url = f"{peer}/api/bonfire/kelly"
                    payload = json.dumps({"kelly": local_k, "p_success": local_p}).encode('utf-8')
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={"Content-Type": "application/json", "User-Agent": "BonfireRingNode/1.9"}
                    )
                    with urllib.request.urlopen(req, timeout=1.0) as response:
                        res_data = json.loads(response.read().decode('utf-8'))
                        kellies.append(res_data.get("kelly", 0.5))
                        successes.append(res_data.get("p_success", 0.5))
                except Exception:
                    pass

            # Update cache variables
            with self.lock:
                self.healthy_peers = active_healthy
                self.cached_signatures = sigs
                if kellies:
                    self.cached_consensus_kelly = sum(kellies) / len(kellies)
                if successes:
                    self.cached_consensus_p_success = sum(successes) / len(successes)

    def synchronize_invariants(self, local_signature: str) -> List[str]:
        """Non-blocking: Returns the last cached peer signatures and updates local signature."""
        with self.lock:
            self.local_signature = local_signature
            return list(self.cached_signatures.values())

    def share_kelly_consensus(self, local_kelly: float, local_p_success: float) -> Dict[str, float]:
        """Non-blocking: Returns the last cached consensus values and updates local parameters."""
        with self.lock:
            self.local_kelly = local_kelly
            self.local_p_success = local_p_success
            return {
                "consensus_kelly": self.cached_consensus_kelly,
                "consensus_p_success": self.cached_consensus_p_success
            }

    def distribute_admr_to_synthetic_endpoint(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Offloads ADMR calculations to a healthy peer if available.
        Uses a short, strict timeout to avoid blocking if the endpoint is unresponsive.
        """
        with self.lock:
            targets = list(self.healthy_peers)
            
        if not targets:
            return None
            
        state_list = state.view(-1).tolist()
        peer = targets[0]
        try:
            url = f"{peer}/api/bonfire/compute_admr"
            payload = json.dumps({"state": state_list}).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "BonfireRingNode/1.9"}
            )
            # Use strict 1.0 second timeout for offloading to ensure responsiveness
            with urllib.request.urlopen(req, timeout=1.0) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                computed = res_data.get("states")
                if computed:
                    return torch.tensor(computed, dtype=torch.float32, device=state.device).view(state.shape)
        except Exception:
            pass
        return None

    def close(self):
        self.running = False
        self.executor.shutdown(wait=False)
