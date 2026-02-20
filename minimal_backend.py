#!/usr/bin/env python3
"""
Minimal Backend - Simple web server without complex imports.
"""

import http.server
import socketserver
import json
import sys
import os
from urllib.parse import urlparse, parse_qs

# Add paths
sys.path.append('src')
sys.path.append('examples')

class MinimalHandler(http.server.SimpleHTTPRequestHandler):
    """Minimal request handler for basic functionality."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self._serve_chat_interface()
        elif self.path == '/ping':
            self._send_json({'status': 'ok', 'message': 'Minimal backend running'})
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/interact':
            self._handle_chat()
        else:
            self.send_error(404, "Not found")
    
    def _serve_chat_interface(self):
        """Serve a simple chat interface."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Gyroidic AI Chat</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 800px; margin: 0 auto; }
        .chat-box { border: 1px solid #444; height: 400px; overflow-y: scroll; padding: 10px; background: #2a2a2a; margin-bottom: 10px; }
        .input-area { display: flex; }
        .input-area input { flex: 1; padding: 10px; background: #333; color: #fff; border: 1px solid #555; }
        .input-area button { padding: 10px 20px; background: #0066cc; color: #fff; border: none; cursor: pointer; }
        .message { margin: 10px 0; padding: 5px; }
        .user { background: #003366; border-radius: 5px; }
        .ai { background: #330066; border-radius: 5px; }
        .status { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Gyroidic AI Chat Interface</h1>
        <div class="status">✅ Minimal backend running - Basic functionality available</div>
        
        <div id="chat-box" class="chat-box">
            <div class="message ai">
                <strong>AI:</strong> Hello! I'm a minimal version of the Gyroidic AI system. 
                The full backend had import issues, so this is a simplified interface. 
                I can respond to basic queries, but advanced features are limited.
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        function addMessage(sender, message) {
            const chatBox = document.getElementById('chat-box');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender.toLowerCase();
            messageDiv.innerHTML = '<strong>' + sender + ':</strong> ' + message;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;

            addMessage('User', message);
            input.value = '';

            fetch('/interact', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: message})
            })
            .then(response => response.json())
            .then(data => {
                addMessage('AI', data.response || 'Sorry, I had trouble processing that.');
            })
            .catch(error => {
                addMessage('AI', 'Error: Could not connect to backend.');
            });
        }
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _handle_chat(self):
        """Handle chat interactions."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            user_text = data.get('text', '').strip()
            
            # Simple response logic (no complex AI for now)
            response = self._generate_simple_response(user_text)
            
            self._send_json({
                'response': response,
                'status': 'success',
                'backend': 'minimal'
            })
            
        except Exception as e:
            self._send_json({
                'response': f'Error processing request: {str(e)}',
                'status': 'error'
            })
    
    def _generate_simple_response(self, text):
        """Generate a simple response without complex AI."""
        text_lower = text.lower()
        
        if 'hello' in text_lower or 'hi' in text_lower:
            return "Hello! I'm the minimal Gyroidic AI. The full system had import issues, but I can still chat with you."
        
        elif 'how are you' in text_lower:
            return "I'm running in minimal mode due to import conflicts in the full backend. But I'm functional!"
        
        elif 'test' in text_lower:
            return "✅ Minimal backend test successful! The full system components work individually, but the integrated backend has import issues."
        
        elif 'help' in text_lower:
            return """Available commands:
• Individual tests work: python test_image_simple.py
• Temporal training: python examples/enhanced_temporal_training.py  
• Mandelbulb augmentation: python test_mandelbulb_simple.py
• The full backend has Python import issues that need fixing."""
        
        elif 'status' in text_lower:
            return "🔧 Status: Minimal backend running. Core AI components work individually but integration has import conflicts."
        
        else:
            return f"I received: '{text}'. I'm in minimal mode, so my responses are basic. The full AI system needs import fixes to work properly."
    
    def _send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def main():
    """Start the minimal backend server."""
    port = 8000
    
    print("🚀 Minimal Gyroidic Backend")
    print("=" * 30)
    print(f"🌐 Starting server on port {port}...")
    
    try:
        with socketserver.TCPServer(("", port), MinimalHandler) as httpd:
            print(f"✅ Server running at http://localhost:{port}")
            print("🔧 This is a minimal version due to import issues in the full backend")
            print("⏹️  Press Ctrl+C to stop")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()