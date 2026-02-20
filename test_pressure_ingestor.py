#!/usr/bin/env python3
"""
Test the Pressure Ingestor - Runtime code generation for constraint forcing.
"""

import sys
import os
sys.path.append('src')

from data.pressure_ingestor import PressureIngestor

def test_pressure_ingestor():
    """Test the pressure ingestor system."""
    print("🔥 Testing Pressure Ingestor - Runtime Code Generation")
    print("=" * 60)
    
    # Initialize ingestor
    ingestor = PressureIngestor(device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu')
    
    print(f"📋 Available sources: {list(ingestor.sources.keys())}")
    
    # Test single source materialization
    print(f"\n🎯 Testing Single Source Materialization")
    print("-" * 40)
    
    # Test OEIS bulk (most likely to succeed)
    result = ingestor.materialize_source('oeis_bulk')
    
    print(f"\n📊 OEIS Bulk Result:")
    print(f"   • Source: {result['source']}")
    print(f"   • Constraints ready: {result['constraints_ready']}")
    print(f"   • Final state: {result['final_state']}")
    
    # Show phase results
    for phase, phase_result in result['phase_results'].items():
        if 'error' in phase_result:
            print(f"   • {phase}: ❌ {phase_result['error']}")
        else:
            print(f"   • {phase}: ✅ Success")
            if phase == 'verify' and 'constraints_extracted' in phase_result:
                print(f"     - Constraints: {phase_result['constraints_extracted']}")
                print(f"     - Collisions: {phase_result['collision_count']}")
    
    # Test pressure ingestion across multiple sources
    print(f"\n🔥 Testing Multi-Source Pressure Ingestion")
    print("-" * 40)
    
    # Test with sources most likely to work
    test_sources = ['oeis_bulk', 'debian_sources']
    
    pressure_report = ingestor.force_pressure_ingestion(test_sources)
    
    print(f"\n📊 Pressure Report Summary:")
    print(f"   • Sources attempted: {pressure_report['total_sources_attempted']}")
    print(f"   • Sources materialized: {pressure_report['sources_materialized']}")
    print(f"   • Total constraints: {pressure_report['total_constraints_extracted']}")
    print(f"   • Total collisions: {pressure_report['total_collisions_detected']}")
    print(f"   • Pressure density: {pressure_report['pressure_density']:.3f}")
    print(f"   • Rigidity variance: {pressure_report['rigidity_variance']:.3f}")
    
    # Test constraint batch generation
    print(f"\n🔧 Testing Constraint Batch Generation")
    print("-" * 40)
    
    constraint_batch = ingestor.get_constraint_batch(batch_size=8)
    
    print(f"📊 Generated Constraint Batch:")
    print(f"   • Batch shape: {constraint_batch.shape}")
    print(f"   • Value range: [{constraint_batch.min():.3f}, {constraint_batch.max():.3f}]")
    print(f"   • Batch variance: {constraint_batch.var():.3f}")
    
    # Test code generation caching
    print(f"\n🔧 Testing Code Generation Caching")
    print("-" * 40)
    
    print(f"📊 Generated Code Cache:")
    for cache_key, code_snippet in ingestor.generated_code_cache.items():
        print(f"   • {cache_key}: {len(code_snippet)} chars")
    
    # Show sample generated code
    if ingestor.generated_code_cache:
        sample_key = list(ingestor.generated_code_cache.keys())[0]
        sample_code = ingestor.generated_code_cache[sample_key]
        print(f"\n📝 Sample Generated Code ({sample_key}):")
        print("-" * 30)
        print(sample_code[:300] + "..." if len(sample_code) > 300 else sample_code)
    
    print(f"\n🎉 Pressure Ingestor Test Complete!")
    print("=" * 60)
    
    # Assessment
    if pressure_report['sources_materialized'] > 0:
        print("✅ SUCCESS: At least one source materialized")
        if pressure_report['pressure_density'] > 0.1:
            print("🔥 HIGH IMPACT: Significant constraint pressure generated")
        else:
            print("⚡ MEDIUM IMPACT: Some constraint pressure generated")
    else:
        print("⚠️  LIMITED SUCCESS: No sources fully materialized")
        print("💡 This is expected for bulk sources without proper credentials/access")
    
    print(f"\n🎯 Key Advantages of This Approach:")
    print("   • Runtime code generation keeps token count low")
    print("   • Failure-first execution reveals structural issues")
    print("   • No polite API assumptions - bulk or nothing")
    print("   • State transitions without reasoning overhead")
    print("   • Constraint pressure measurement, not optimization")

if __name__ == "__main__":
    test_pressure_ingestor()
