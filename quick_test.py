#!/usr/bin/env python3
import requests
import json

try:
    print("🧪 Quick Phase 2.1 Test")
    response = requests.post(
        'http://localhost:8000/interact',
        json={'text': 'hello'},
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS!")
        print(f"📥 Output: '{data['response']}'")
        print(f"📊 Length: {len(data['response'])}")
        print(f"🔧 Repair diagnostics: {'repair_diagnostics' in data}")
        
        if 'repair_diagnostics' in data:
            print(f"🔧 Components: {list(data['repair_diagnostics'].keys())}")
    else:
        print(f"❌ Failed: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")