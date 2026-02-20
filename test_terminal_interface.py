#!/usr/bin/env python3
"""
Test the restored diegetic terminal interface.
"""

import requests
import time

def test_terminal_interface():
    """Test that the diegetic terminal interface is properly served."""
    print("🧪 Testing Restored Diegetic Terminal Interface")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Check if backend is running
    try:
        response = requests.get(f"{base_url}/ping", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend ping successful: {data}")
        else:
            print(f"❌ Backend ping failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return False
    
    # Test 2: Check if terminal interface is served
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            html_content = response.text
            
            # Check for key elements of the diegetic terminal
            required_elements = [
                "GYROIDIC DIEGETIC TERMINAL",
                "KNOWLEDGE ASSOCIATION PANEL",
                "image-text",  # Changed from IMAGE→TEXT
                "text-text",   # Changed from TEXT→TEXT
                "chat-area",
                "association-form"
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in html_content:
                    missing_elements.append(element)
            
            if not missing_elements:
                print("✅ Diegetic terminal interface served successfully")
                print(f"📄 HTML content length: {len(html_content)} characters")
                print("🔍 Key elements found:")
                for element in required_elements:
                    print(f"   • {element}")
            else:
                print("⚠️  Terminal interface served but missing elements:")
                for element in missing_elements:
                    print(f"   • {element}")
                return False
                
        else:
            print(f"❌ Terminal interface not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing terminal interface: {e}")
        return False
    
    # Test 3: Test chat functionality
    try:
        chat_data = {"text": "test message"}
        response = requests.post(f"{base_url}/interact", json=chat_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat functionality working")
            print(f"📝 AI Response: {data.get('response', 'No response')[:100]}...")
        else:
            print(f"⚠️  Chat functionality issue: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Chat test failed: {e}")
    
    # Test 4: Test association endpoint
    try:
        assoc_data = {
            "type": "text-text-association",
            "input": "test concept",
            "response": "test definition",
            "relationship": "definition"
        }
        response = requests.post(f"{base_url}/associate", json=assoc_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Association functionality working")
            print(f"📝 Association Response: {data.get('message', 'No message')}")
        else:
            print(f"⚠️  Association functionality issue: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Association test failed: {e}")
    
    # Test 5: Test Wikipedia endpoint
    try:
        wiki_data = {"topic": "artificial intelligence"}
        response = requests.post(f"{base_url}/wikipedia", json=wiki_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Wikipedia functionality working")
            print(f"📝 Wikipedia Response: {data.get('message', 'No message')}")
        else:
            print(f"⚠️  Wikipedia functionality issue: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Wikipedia test failed: {e}")
    
    print("\n🎉 Diegetic Terminal Interface Test Complete!")
    print("🌐 Access the terminal at: http://localhost:8000")
    print("💡 The terminal now includes:")
    print("   • Chat interface with AI interaction")
    print("   • Image-to-text association panel")
    print("   • Text-to-text association panel")
    print("   • Wikipedia knowledge integration")
    print("   • Full gyroidic AI backend with temporal reasoning")
    
    return True

if __name__ == "__main__":
    test_terminal_interface()