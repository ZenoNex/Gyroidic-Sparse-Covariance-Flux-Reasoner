
import requests
import time
import subprocess
import sys
import os
import threading
import json

# Define the backend script path
BACKEND_SCRIPT = "hybrid_backend.py"
PORT = 8080
HOST = "127.0.0.1"

def test_local_mode():
    url = f"http://{HOST}:{PORT}/api/test_token"
    payload = {"token": "LOCAL_MODE"}
    
    print(f"Testing POST {url} with LOCAL_MODE token...")
    try:
        response = requests.post(url, json=payload, timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("Response JSON:", data)
                if data.get("success") and data.get("username") == "Local User":
                    print("SUCCESS: Local Mode token accepted.")
                else:
                    print("FAILURE: Local Mode token rejected or incorrect response.")
            except ValueError:
                print("FAILURE: Invalid JSON response.")
        else:
            print(f"FAILURE: Backend returned status {response.status_code}")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")

def stream_output(proc):
    for line in iter(proc.stdout.readline, ''):
        print(f"[BACKEND] {line.strip()}")
    proc.stdout.close()

if __name__ == "__main__":
    proc = None
    try:
        print(f"Checking if backend is running on {HOST}:{PORT}...")
        try:
            requests.get(f"http://{HOST}:{PORT}/ping", timeout=2)
            print("Backend already running. Proceeding with tests...")
            test_local_mode()
        except:
            print("Backend not running. Starting it...")
            
            # Start backend in background with unbuffered output and utf-8 encoding
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            cmd = [sys.executable, "-u", BACKEND_SCRIPT, str(PORT)]
            print(f"Executing: {' '.join(cmd)}")
            
            proc = subprocess.Popen(
                cmd, 
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            # Start output reading thread
            t = threading.Thread(target=stream_output, args=(proc,), daemon=True)
            t.start()
            
            print("Waiting 15 seconds for startup...")
            time.sleep(15) 
            
            if proc.poll() is not None:
                print(f"Backend process terminated early with code {proc.returncode}")
            else:
                try:
                    test_local_mode()
                except Exception as e:
                    print(f"Test failed: {e}")
                finally:
                    print("Stopping backend...")
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
                    
    except Exception as e:
        print(f"Script failed: {e}")
