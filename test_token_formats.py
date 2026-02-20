#!/usr/bin/env python3
"""
Test Different Token Formats

Tests various ways to format the HF token to see which works.
"""

import requests

def test_token_formats(token):
    """Test different token formats."""
    print("=" * 50)
    print("TESTING TOKEN FORMATS")
    print("=" * 50)
    
    print(f"Token: {token[:15]}...{token[-10:]}")
    
    # Different header formats to try
    formats = [
        ("Bearer format", {'Authorization': f'Bearer {token}'}),
        ("Token format", {'Authorization': f'token {token}'}),
        ("Direct format", {'Authorization': token}),
        ("HF format", {'Authorization': f'hf {token}'}),
        ("With User-Agent", {
            'Authorization': f'Bearer {token}',
            'User-Agent': 'python-requests/2.28.0'
        }),
        ("With Accept", {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }),
        ("Full headers", {
            'Authorization': f'Bearer {token}',
            'User-Agent': 'python-requests/2.28.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    ]
    
    for name, headers in formats:
        print(f"\n--- {name} ---")
        print(f"Headers: {headers}")
        
        try:
            response = requests.get('https://huggingface.co/api/whoami', 
                                  headers=headers, timeout=10)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ SUCCESS! User: {data.get('name', 'Unknown')}")
                print(f"✓ This format works!")
                return True
            else:
                print(f"✗ Failed: {response.text[:100]}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return False

def test_with_huggingface_hub(token):
    """Test with huggingface_hub library."""
    print(f"\n--- Testing with huggingface_hub library ---")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        user_info = api.whoami(token=token)
        
        print(f"✓ HfApi SUCCESS! User: {user_info.get('name', 'Unknown')}")
        return True
        
    except Exception as e:
        print(f"✗ HfApi failed: {e}")
        return False

def main():
    """Main test."""
    token = input("Enter your HF token: ").strip()
    
    if not token:
        print("No token provided!")
        return
    
    # Test different formats
    success = test_token_formats(token)
    
    if not success:
        # Try with huggingface_hub
        hub_success = test_with_huggingface_hub(token)
        
        if hub_success:
            print(f"\n✓ The huggingface_hub library works!")
            print(f"The issue is with our manual API calls.")
        else:
            print(f"\n✗ Even huggingface_hub fails!")
            print(f"There might be an issue with the token itself.")
    
    print(f"\nIf nothing works, try:")
    print(f"1. huggingface-cli login")
    print(f"2. Create a completely new token")
    print(f"3. Check if your account has any restrictions")

if __name__ == "__main__":
    main()