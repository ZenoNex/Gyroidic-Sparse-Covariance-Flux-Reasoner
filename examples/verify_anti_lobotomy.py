"""
Verify Constitutional Alignment: Anti-Lobotomy Logic
Tests that NarrativeCollapseDetector correctly flags Homological Collapse (Lobotomy).
"""
import sys
import os
import torch
import warnings

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.narrative_collapse import NarrativeCollapseDetector

def test_anti_lobotomy_detection():
    print("=" * 60)
    print("Verifying Constitutional Alignment: Anti-Lobotomy Logic")
    print("=" * 60)
    
    # Initialize detector
    detector = NarrativeCollapseDetector(lobotomy_threshold=0.5)
    
    # Mock data
    # 1. Healthy State: High Complexity (High Betti)
    print("\n[1] Initial State (High Complexity)...")
    # Mock residues/manifold that produce a complex graph
    # We'll just mock the internal state of the monitor for this test 
    # since constructing actual homology inputs is complex.
    # We act "as if" the monitor calculated Betti = 10
    detector.last_betti_count = 10.0
    print("    Current Betti: 10.0")
    
    # 2. Lobotomy Event: Complexity drops, Gradient is LOW (Calm)
    print("\n[2] Simulating Lobotomy (Betti Drop + Calm System)...")
    
    # Mocking the internal PAS monitor call result
    # We'll override the method momentarily for testing logic flow
    original_monitor = detector.pas_monitor
    
    class MockMonitor:
        def build_graph(self, residues, manifold):
            return MockGraph(num_edges=5, num_nodes=5) # Betti = 5 - 5 + 1 = 1
            
    class MockGraph:
        def __init__(self, num_edges, num_nodes):
            self._edges = num_edges
            self._nodes = num_nodes
        def number_of_edges(self): return self._edges
        def number_of_nodes(self): return self._nodes
        
    detector.pas_monitor = MockMonitor()
    
    # Inputs (Dont matter much with mocked monitor)
    residues = [torch.randn(10, 3)]
    manifold = torch.randn(10, 3)
    
    # Trigger check with LOW GRADIENT (0.01)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        is_lobotomized, delta = detector.detect_homological_collapse(
            residues, manifold, current_gradient_norm=0.01
        )
        
        print(f"    Betti Delta: {delta}")
        print(f"    Gradient: 0.01")
        print(f"    Is Lobotomized: {is_lobotomized}")
        
        if is_lobotomized and "CONSTITUTIONAL ALARM" in str(w[-1].message):
            print("    ✓ PASSED: Lobotomy detected (Low Gradient + Complexity Drop)")
        else:
            print(f"    ✗ FAILED: Should detect lobotomy! Warnings: {[str(x.message) for x in w]}")

    # 3. Valid Resolution: Complexity drops, Gradient is HIGH (Struggle)
    print("\n[3] Simulating Valid Resolution (Betti Drop + High Gradient)...")
    detector.last_betti_count = 10.0 # Reset
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Gradient > Threshold (0.5)
        is_lobotomized, delta = detector.detect_homological_collapse(
            residues, manifold, current_gradient_norm=1.0
        )
        
        print(f"    Betti Delta: {delta}")
        print(f"    Gradient: 1.0")
        print(f"    Is Lobotomized: {is_lobotomized}")
        
        if not is_lobotomized:
            print("    ✓ PASSED: Resolution accepted (High Gradient + Complexity Drop)")
        else:
            print("    ✗ FAILED: Should allow resolution when gradient is high!")

    print("\n" + "=" * 60)
    
    # Restore
    detector.pas_monitor = original_monitor
    return True

if __name__ == "__main__":
    test_anti_lobotomy_detection()
