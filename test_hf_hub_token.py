#!/usr/bin/env python3
"""
Test HF Token using huggingface_hub

Uses the official huggingface_hub library to test token validity.
"""

import os
from huggingface_hub import HfApi, login, whoami
from huggingface_hub.utils import HfHubHTTPError

def test_token_with_hub():
    """Test token using huggingface_hub library."""
    print("=" * 50)
    print("HUGGING FACE HUB TOKEN TEST")
    print("=" * 50)
    
    # Get token from user
    token = input("Enter your HF token: ").strip()
    
    if not token:
        print("No token provided!")
        return False
    
    print(f"Testing token: {token[:10]}...{token[-5:]}")
    
    try:
        # Method 1: Direct API test
        print("\n1. Testing with HfApi...")
        api = HfApi(token=token)
        
        # Test whoami
        user_info = api.whoami(token=token)
        print(f"   SUCCESS: User = {user_info['name']}")
        print(f"   Type: {user_info.get('type', 'Unknown')}")
        print(f"   Email verified: {user_info.get('emailVerified', 'Unknown')}")
        
        # Method 2: Test login
        print("\n2. Testing login...")
        login(token=token, add_to_git_credential=False)
        print("   SUCCESS: Login worked!")
        
        # Method 3: Test dataset access
        print("\n3. Testing dataset access...")
        
        datasets_to_test = [
            'lmsys/lmsys-chat-1m',
            'OpenAssistant/oasst2',
            'squad'  # Public dataset
        ]
        
        for dataset_id in datasets_to_test:
            try:
                dataset_info = api.dataset_info(dataset_id, token=token)
                print(f"   ✓ {dataset_id}: Accessible")
                print(f"     Downloads: {dataset_info.downloads:,}")
                print(f"     Private: {dataset_info.private}")
                print(f"     Gated: {getattr(dataset_info, 'gated', 'Unknown')}")
            except HfHubHTTPError as e:
                if e.response.status_code == 403:
                    print(f"   ⚠ {dataset_id}: Gated (need agreement)")
                elif e.response.status_code == 404:
                    print(f"   ✗ {dataset_id}: Not found")
                else:
                    print(f"   ✗ {dataset_id}: Error {e.response.status_code}")
            except Exception as e:
                print(f"   ✗ {dataset_id}: {str(e)[:50]}...")
        
        # Method 4: Test datasets library integration
        print("\n4. Testing datasets library integration...")
        
        # Set environment variables
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
        
        try:
            from datasets import load_dataset_builder
            
            # Test with a public dataset
            builder = load_dataset_builder("squad")
            print(f"   ✓ Datasets library works with squad")
            
            # Test with gated dataset
            try:
                builder = load_dataset_builder("lmsys/lmsys-chat-1m", use_auth_token=token)
                print(f"   ✓ Datasets library works with lmsys-chat-1m")
            except Exception as e:
                print(f"   ⚠ lmsys-chat-1m: {str(e)[:50]}...")
                
        except ImportError:
            print("   ⚠ datasets library not installed")
        except Exception as e:
            print(f"   ✗ datasets library error: {str(e)[:50]}...")
        
        print(f"\n" + "=" * 50)
        print("TOKEN IS VALID AND WORKING!")
        print("=" * 50)
        
        return True
        
    except HfHubHTTPError as e:
        print(f"\n✗ HTTP Error {e.response.status_code}")
        if e.response.status_code == 401:
            print("   Token is invalid or expired")
            print("   Check: https://huggingface.co/settings/tokens")
        elif e.response.status_code == 403:
            print("   Token has insufficient permissions")
            print("   Make sure token has 'Read' permissions")
        else:
            print(f"   Response: {e.response.text[:200]}")
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def main():
    """Main function."""
    success = test_token_with_hub()
    
    if success:
        print("\n🎉 Your token works perfectly!")
        print("You can now:")
        print("1. Run: python launch_conversational_gui.py")
        print("2. Enter your token in the GUI")
        print("3. Start training!")
    else:
        print("\n❌ Token issues detected")
        print("Try:")
        print("1. huggingface-cli login")
        print("2. Create a new token at https://huggingface.co/settings/tokens")
        print("3. Make sure token has 'Read' permissions")

if __name__ == "__main__":
    main()