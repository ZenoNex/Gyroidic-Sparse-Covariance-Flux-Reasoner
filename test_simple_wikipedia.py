#!/usr/bin/env python3
"""
Simple Wikipedia System Test
Quick test to verify basic functionality.
"""

import requests
import json
import time

def test_backend_ping():
    """Test that backend is responding."""
    try:
        response = requests.get('http://localhost:8000/ping', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend online: PID {data.get('pid')}")
            return True
        else:
            print(f"❌ Backend error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False

def test_html_page():
    """Test that HTML page loads."""
    try:
        response = requests.get('http://localhost:8000/wikipedia-trainer', timeout=10)
        if response.status_code == 200:
            content = response.text
            print(f"📄 HTML content length: {len(content)}")
            print(f"📝 First 200 chars: {repr(content[:200])}")
            
            if 'Wikipedia Knowledge Ingestion System' in content:
                print("✅ HTML page loads with correct title")
                return True
            else:
                print("❌ HTML page missing title")
                # Check for other key elements
                if 'Content Input' in content:
                    print("✅ Content Input found")
                if 'Extraction Options' in content:
                    print("✅ Extraction Options found")
                if 'Training Status' in content:
                    print("✅ Training Status found")
                return False
        else:
            print(f"❌ HTML page error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ HTML page failed: {e}")
        return False

def test_basic_association():
    """Test basic association functionality."""
    try:
        response = requests.post(
            'http://localhost:8000/associate',
            json={
                'source': 'test concept',
                'target': 'test content for association learning'
            },
            timeout=15
        )
        
        if response.status_code == 200:
            print("✅ Basic association working")
            return True
        else:
            print(f"❌ Association error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Association failed: {e}")
        return False

def main():
    print("🧪 Simple Wikipedia System Test")
    print("=" * 40)
    
    tests = [
        ("Backend Ping", test_backend_ping),
        ("HTML Page", test_html_page),
        ("Basic Association", test_basic_association)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔧 Testing {test_name}...")
        success = test_func()
        results.append((test_name, success))
    
    print(f"\n{'='*40}")
    print("📊 Results:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"• {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Basic system working!")
        print("🌐 Open: http://localhost:8000/wikipedia-trainer")
    else:
        print("⚠️  Some issues detected")

if __name__ == "__main__":
    main()