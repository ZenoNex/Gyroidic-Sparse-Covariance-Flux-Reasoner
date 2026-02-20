#!/usr/bin/env python3
"""
Debug API Endpoint

Check if we're hitting the right HuggingFace API endpoint.
"""

import requests
import json

def test_api_endpoints(token):
    """Test different HuggingFace API endpoints."""
    print("=" * 60)
    print("TESTING HUGGINGFACE API ENDPOINTS")
    print("=" * 60)
    
    print(f"Token: {token[:15]}...{token[-10:]}")
    
    # Different endpoints to try
    endpoints = [
        ("Current whoami", "https://huggingface.co/api/whoami"),
        ("Hub API whoami", "https://hub-api.huggingface.co/whoami"),
        ("API v1 whoami", "https://huggingface.co/api/v1/whoami"),
        ("User info", "https://huggingface.co/api/user"),
        ("Models endpoint", "https://huggingface.co/api/models?limit=1"),
    ]
    
    headers = {'Authorization': f'Bearer {token}'}
    
    for name, url in endpoints:
        print(f"\n--- {name} ---")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"✓ SUCCESS!")
                    print(f"Response: {json.dumps(data, indent=2)[:200]}...")
                    return True
                except:
                    print(f"✓ SUCCESS (non-JSON): {response.text[:100]}...")
                    return True
            else:
                print(f"✗ Failed")
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return False

def test_with_curl_simulation(token):
    """Simulate what curl would do."""
    print(f"\n--- Simulating curl command ---")
    
    # This is what should work according to HF docs
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'curl/7.68.0',
        'Accept': '*/*'
    }
    
    try:
        response = requests.get('https://huggingface.co/api/whoami', 
                              headers=headers, timeout=15)
        
        print(f"Curl simulation status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Curl simulation SUCCESS! User: {data.get('name', 'Unknown')}")
            return True
        else:
            print(f"✗ Curl simulation failed: {response.text[:200]}")
            
    except Exception as e:
        print(f"✗ Curl simulation error: {e}")
    
    return False

def test_huggingface_hub_library(token):
    """Test with the official library."""
    print(f"\n--- Testing with huggingface_hub library ---")
    
    try:
        from huggingface_hub import HfApi, login
        
        # Method 1: Direct API
        api = HfApi(token=token)
        user_info = api.whoami(token=token)
        
        print(f"✓ HfApi SUCCESS! User: {user_info.get('name', 'Unknown')}")
        print(f"Full info: {json.dumps(user_info, indent=2)}")
        return True
        
    except ImportError:
        print("✗ huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"✗ HfApi failed: {e}")
        return False

def main():
    """Main debugging."""
    print("Debugging HuggingFace API Access")
    print("This will test different endpoints and methods")
    
    token = input("\nEnter your HF token: ").strip()
    
    if not token:
        print("No token provided!")
        return
    
    # Test different endpoints
    endpoint_success = test_api_endpoints(token)
    
    # Test curl simulation
    curl_success = test_with_curl_simulation(token)
    
    # Test official library
    lib_success = test_huggingface_hub_library(token)
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Endpoint tests: {'✓' if endpoint_success else '✗'}")
    print(f"Curl simulation: {'✓' if curl_success else '✗'}")
    print(f"Official library: {'✓' if lib_success else '✗'}")
    
    if lib_success and not endpoint_success:
        print(f"\n💡 The official library works but our API calls don't!")
        print(f"This suggests we need to use the huggingface_hub library instead of direct API calls.")
    elif not any([endpoint_success, curl_success, lib_success]):
        print(f"\n❌ Nothing works - there might be an account or token issue.")
    else:
        print(f"\n✓ At least one method works!")

if __name__ == "__main__":
    main()