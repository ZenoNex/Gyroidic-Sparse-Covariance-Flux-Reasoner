
import requests
import json

def test_backend_spaces():
    url = "http://localhost:8000/interact"
    payload = {
        "text": "INGEST_DYAD: [0.1] | A test for spaces"
    }
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response Body: '{response.text}'")
        
        data = response.json()
        resp_text = data.get('response', '')
        print(f"Extracted response: '{resp_text}'")
        if " " in resp_text:
            print("[OK] Spaces preserved in backend response.")
        else:
            print("[FAIL] Spaces stripped in backend response.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_backend_spaces()
