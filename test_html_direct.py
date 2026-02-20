#!/usr/bin/env python3
"""
Direct HTML serving test to isolate the issue.
"""

import http.server
import socketserver
import os
import threading
import time
import requests

class SimpleHTMLHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/test-html':
            try:
                # Try multiple paths
                paths_to_try = [
                    'src/ui/wikipedia_trainer.html',
                    os.path.join('src', 'ui', 'wikipedia_trainer.html'),
                    os.path.join(os.getcwd(), 'src', 'ui', 'wikipedia_trainer.html')
                ]
                
                content = None
                used_path = None
                
                for path in paths_to_try:
                    print(f"🔧 Trying path: {path}")
                    print(f"🔧 Exists: {os.path.exists(path)}")
                    if os.path.exists(path):
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            used_path = path
                            print(f"✅ Successfully read from {path}")
                            break
                        except Exception as e:
                            print(f"❌ Error reading {path}: {e}")
                            import traceback
                            traceback.print_exc()
                
                if content is not None and len(content) > 0:
                    print(f"✅ Successfully read HTML from: {used_path}")
                    print(f"📊 Content length: {len(content)}")
                    print(f"📝 First 100 chars: {repr(content[:100])}")
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                else:
                    print(f"❌ Content is empty or None: {repr(content)}")
                    self.send_error(404, "HTML file is empty")
                    
            except Exception as e:
                print(f"❌ Exception in HTML serving: {e}")
                import traceback
                traceback.print_exc()
                self.send_error(500, str(e))
        else:
            self.send_error(404)

def test_direct_html():
    print("🧪 Direct HTML Serving Test")
    print("=" * 40)
    
    # Start simple server
    port = 8001
    httpd = socketserver.TCPServer(("", port), SimpleHTMLHandler)
    
    # Run server in background
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    print(f"🚀 Test server started on port {port}")
    time.sleep(1)
    
    try:
        # Test the HTML serving
        response = requests.get(f'http://localhost:{port}/test-html', timeout=10)
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response length: {len(response.text)}")
        
        if response.status_code == 200 and len(response.text) > 0:
            print("✅ Direct HTML serving works!")
            if 'Wikipedia Knowledge Ingestion System' in response.text:
                print("✅ Title found in content")
            else:
                print("⚠️  Title not found in content")
        else:
            print("❌ Direct HTML serving failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        httpd.shutdown()
        httpd.server_close()

if __name__ == "__main__":
    test_direct_html()