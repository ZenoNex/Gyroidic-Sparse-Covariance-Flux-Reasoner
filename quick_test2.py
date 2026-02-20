#!/usr/bin/env python3
import requests
import json

try:
    print("🧪 Phase 2.3 Test (Chern-Simons Gasket)")
    response = requests.post(
        'http://localhost:8000/interact',
        json={'text': 'test'},
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS!")
        print(f"📥 Output: '{data['response']}'")
        print(f"📊 Length: {len(data['response'])}")
        
        if 'repair_diagnostics' in data:
            components = list(data['repair_diagnostics'].keys())
            print(f"🔧 Active components: {components}")
            
            # Check if Chern-Simons is working
            if 'chern_simons_gasket' in data['repair_diagnostics']:
                cs_diag = data['repair_diagnostics']['chern_simons_gasket']
                print(f"🔍 Chern-Simons status: {cs_diag}")
            else:
                print("⚠️  Chern-Simons Gasket not found in diagnostics")
        else:
            print("❌ No repair diagnostics found")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")