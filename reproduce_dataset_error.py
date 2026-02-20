
import requests
import time
import subprocess
import sys
import os
import threading

# Define the backend script path
BACKEND_SCRIPT = "hybrid_backend.py"
PORT = 8080
HOST = "127.0.0.1"

def test_routes():
    base_url = f"http://{HOST}:{PORT}"
    
    print(f"Testing routes on {base_url}...")
    
    # Test 1: local_datasets (GET)
    try:
        print(f"Testing GET {base_url}/api/local_datasets...")
        response = requests.get(f"{base_url}/api/local_datasets", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                print("Response JSON:", data)
                if data.get("success"):
                    print("SUCCESS: /api/local_datasets working correctly.")
                else:
                    print("FAILURE: /api/local_datasets returned success=False.")
            except ValueError:
                print("FAILURE: Response is not valid JSON (likely HTML 404/Error page).")
        else:
            print(f"FAILURE: Failed with status {response.status_code}")
    except Exception as e:
        print(f"EXCEPTION testing local_datasets: {e}")

    # Test 2: ingest_local (POST) - Dry run or invalid to check route existence
    try:
        print(f"\nTesting POST {base_url}/api/ingest_local...")
        payload = {"dataset": "test_dummy", "max_samples": 1, "quality_threshold": 0.1}
        response = requests.post(f"{base_url}/api/ingest_local", json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                print("Response JSON:", data)
                 # Expecting failure on dummy dataset, but route should exist and return JSON
                if "success" in data: 
                    print("SUCCESS: /api/ingest_local route exists and returned JSON.")
                # Also accept success=False with specific message
                elif data.get("message") == "Ingestion failed check logs" or "Dataset system not initialized" in data.get("message", ""):
                    print("SUCCESS: /api/ingest_local route exists (returned expected failure message).")
                else:
                    print(f"FAILURE: /api/ingest_local returned unexpected JSON structure: {data}")
            except ValueError:
                 print("FAILURE: Response is not valid JSON (likely HTML 404/Error page).")
        elif response.status_code == 404:
             print("FAILURE: Route /api/ingest_local not found (404).")
        else:
            print(f"WARNING: Failed with status {response.status_code}")

    except Exception as e:
        print(f"EXCEPTION testing ingest_local: {e}")

def stream_output(proc):
    for line in iter(proc.stdout.readline, ''):
        print(f"[BACKEND] {line.strip()}")
    proc.stdout.close()

if __name__ == "__main__":
    proc = None
    try:
        print(f"Checking if backend is running on {HOST}:{PORT}...")
        requests.get(f"http://{HOST}:{PORT}/ping", timeout=2)
        print("Backend appears to be running. Proceeding with tests...")
        test_routes()
    except:
        print("Backend not running or not reachable. Attempting to start it for testing...")
        
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
                test_routes()
            except Exception as e:
                print(f"Test failed: {e}")
            finally:
                print("Stopping backend...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
