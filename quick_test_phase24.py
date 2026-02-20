#!/usr/bin/env python3
import requests
import json

try:
    print("🧪 Phase 2.4 Test (Soliton Stability Healer)")
    response = requests.post(
        'http://localhost:8000/interact',
        json={'text': 'phase24'},
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
            
            # Check if all 4 components are working
            expected_components = [
                'spectral_coherence_corrector',
                'bezout_coefficient_refresh', 
                'chern_simons_gasket',
                'soliton_stability_healer'
            ]
            
            working_components = [comp for comp in expected_components if comp in components]
            print(f"✅ Working components ({len(working_components)}/4): {working_components}")
            
            # Check Soliton Healer specifically
            if 'soliton_stability_healer' in data['repair_diagnostics']:
                soliton_diag = data['repair_diagnostics']['soliton_stability_healer']
                print(f"🔍 Soliton Healer status: {soliton_diag}")
            else:
                print("⚠️  Soliton Stability Healer not found in diagnostics")
        else:
            print("❌ No repair diagnostics found")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")