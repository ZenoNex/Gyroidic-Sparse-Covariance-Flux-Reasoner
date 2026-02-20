#!/usr/bin/env python3
"""
Test backend startup to see specific errors.
"""

try:
    print("🔧 Testing backend import...")
    import sys
    sys.path.append('.')
    
    from src.ui.diegetic_backend import DiegeticPhysicsEngine
    print("✅ Backend import successful")
    
    print("🔧 Testing backend initialization...")
    engine = DiegeticPhysicsEngine()
    print("✅ Backend initialization successful")
    
except Exception as e:
    print(f"❌ Backend startup failed: {e}")
    import traceback
    traceback.print_exc()