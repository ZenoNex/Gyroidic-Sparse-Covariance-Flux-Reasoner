#!/usr/bin/env python3
"""
Simple backend test to isolate the hanging issue.
"""

import http.server
import socketserver
import json
import time

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_GET(self):
        if self.path == '/ping':
            self._send_json({
                "status": "online",
                "pid": 12345,
                "uptime": time.time()
            })
            return
        super().do_GET()

    def do_POST(self):
        if self.path == '/interact':
            print("📥 Received /interact request")
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                user_text = data.get('text', '')
                
                print(f"📝 Processing: '{user_text}'")
                
                # Simple response without repair system
                result = {
                    "response": f"Echo: {user_text}",
                    "iteration": 1,
                    "spectral_entropy": 0.5,
                    "chiral_score": 0.1,
                    "coprime_lock": False
                }
                
                print("✅ Sending response")
                self._send_json(result)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                self._send_error_json(str(e))
        else:
            self.send_error(404)

    def _send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error_json(self, message, code=500):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))

if __name__ == "__main__":
    print("🧪 Starting simple test backend on port 8001...")
    with socketserver.TCPServer(("", 8001), TestHandler) as httpd:
        print("✅ Test backend ready")
        httpd.serve_forever()