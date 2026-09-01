import os
import time
import requests
import logging

class OpenRouterClient:
    def __init__(self, default_model="nousresearch/nous-hermes-2-mixtral-8x7b-dpo"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.default_model = default_model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.latency_ms = 0
        self.last_status = "DISCONNECTED"
        
    def query(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key:
            self.last_status = "ERROR (NO KEY)"
            return "[OpenRouter ERROR: No API Key present in ENV]"
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/ZenoNex/Gyroidic-Sparse-Covariance-Flux-Reasoner",
            "X-Title": "Gyroidic Reasoner Node"
        }
        
        payload = {
            "model": self.default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        
        start_t = time.time()
        try:
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            self.latency_ms = int((time.time() - start_t) * 1000)
            if res.status_code == 200:
                self.last_status = "CONNECTED"
                data = res.json()
                return data["choices"][0]["message"]["content"]
            else:
                self.last_status = f"HTTP {res.status_code}"
                return f"[OpenRouter HTTP {res.status_code}]"
        except Exception as e:
            self.latency_ms = int((time.time() - start_t) * 1000)
            self.last_status = "TIMEOUT/ERROR"
            return f"[OpenRouter Exception: {str(e)}]"

class FederatedNetworkMonitor:
    def __init__(self, freenet_ws=None, freenet_router=None, open_router=None):
        self.freenet_ws = freenet_ws
        self.freenet_router = freenet_router
        self.open_router = open_router
        
    def get_telemetry(self) -> dict:
        telemetry = {
            "freenet_p2p_status": "OFFLINE",
            "freenet_ring_size": 0,
            "openrouter_status": "OFFLINE",
            "openrouter_latency": 0,
            "openrouter_model": "None"
        }
        
        if self.freenet_ws:
            telemetry["freenet_p2p_status"] = "CONNECTED" if getattr(self.freenet_ws, "ws", None) else "DISCONNECTED"
            
        if self.open_router:
            telemetry["openrouter_status"] = self.open_router.last_status
            telemetry["openrouter_latency"] = self.open_router.latency_ms
            telemetry["openrouter_model"] = self.open_router.default_model
            
        return telemetry
