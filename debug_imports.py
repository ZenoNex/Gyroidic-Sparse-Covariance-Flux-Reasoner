#!/usr/bin/env python3
"""
Debug the import chain to find exactly what's failing.
"""

import sys
import os

# Add paths
sys.path.append('src')
sys.path.append('examples')

def test_import(module_name, description):
    """Test importing a specific module."""
    try:
        print(f"🔍 Testing {description}...")
        if module_name == "enhanced_temporal_training":
            from enhanced_temporal_training import NonLobotomyTemporalModel
            print(f"   ✅ {description} - OK")
            return True
        elif module_name == "spectral_coherence_repair":
            from core.spectral_coherence_repair import SpectralCoherenceCorrector
            print(f"   ✅ {description} - OK")
            return True
        elif module_name == "training_modules":
            from training.temporal_association_trainer import TemporalAssociationTrainer
            print(f"   ✅ {description} - OK")
            return True
        elif module_name == "diegetic_components":
            from models.resonance_cavity import ResonanceCavity
            from models.diegetic_heads import ResonanceLarynx
            print(f"   ✅ {description} - OK")
            return True
        else:
            exec(f"import {module_name}")
            print(f"   ✅ {description} - OK")
            return True
    except Exception as e:
        print(f"   ❌ {description} - FAILED: {e}")
        return False

def main():
    """Test all imports systematically."""
    print("🔬 Gyroidic Import Chain Debugger")
    print("=" * 40)
    
    tests = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("enhanced_temporal_training", "Temporal Model"),
        ("spectral_coherence_repair", "Spectral Corrector"),
        ("training_modules", "Training Modules"),
        ("diegetic_components", "Diegetic Components"),
    ]
    
    results = []
    for module, desc in tests:
        result = test_import(module, desc)
        results.append((desc, result))
    
    print(f"\n📊 Import Test Results:")
    print("=" * 25)
    
    passed = 0
    for desc, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {desc}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} imports successful")
    
    if passed == len(results):
        print("\n🎉 ALL IMPORTS WORKING!")
        print("✅ The issue may be in the backend initialization, not imports")
    else:
        print(f"\n⚠️  Import failures detected")
        print("🔧 Focus on fixing the failed imports first")

if __name__ == "__main__":
    main()