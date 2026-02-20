#!/usr/bin/env python3
"""
Minimal HTTP server test
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class MinimalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"📥 GET request to {self.path}")
        if self.path == '/ping':
            print("✅ Processing /ping")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "online", "pid": 12345, "uptime": 100}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)
    
    def do_POST(self):
        print(f"📥 POST request to {self.path}")
        if self.path == '/interact':
            print("✅ Processing /interact")
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body.decode('utf-8'))
            user_text = data.get('text', '')
            print(f"📝 User said: '{user_text}'")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "response": f"Echo: {user_text}",
                "spectral_entropy": 0.5,
                "chiral_score": 0.1,
                "coprime_lock": False
            }
            self.wfile.write(json.dumps(response).encode())
            print("✅ Response sent!")
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        print(f"📥 OPTIONS request to {self.path}")
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8000), MinimalHandler)
    print("🚀 Minimal server running on port 8000")
    server.serve_forever()