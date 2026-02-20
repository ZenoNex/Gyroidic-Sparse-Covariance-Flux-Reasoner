#!/usr/bin/env python3
"""
Test if the diegetic backend is still running.
"""

import requests

def test_backend_status():
    """Test if backend is running."""
    try:
        response = requests.get("http://localhost:8000/ping", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is running: PID {data.get('pid')}, uptime {data.get('uptime', 0):.1f}s")
            return True
        else:
            print(f"❌ Backend responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not responding: {e}")
        return False

if __name__ == "__main__":
    test_backend_status()