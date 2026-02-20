
import torch
import sys
import os

# Adjust path to include src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.invariants import ImplicationInvariant, SelfReferenceAdmissibility

def test_implication_invariant():
    print("Testing ImplicationInvariant...")
    inv = ImplicationInvariant()
    
    # Case 1: Significant Interaction, Significant Implication (Preserved)
    interaction = torch.randn(10, 5) # Significant norm
    implication = torch.randn(10, 5) # Significant norm
    violation, preservation = inv(interaction, implication)
    
    if torch.any(violation > 0):
        print("FAIL: False positive violation detected.")
    else:
        print(f"PASS: Preservation score: {preservation.mean().item()}")
        
    # Case 2: Significant Interaction, Zero Implication (Lobotomy)
    implication_zero = torch.zeros_like(implication)
    violation, preservation = inv(interaction, implication_zero)
    
    if torch.all(violation == 1.0):
        print(f"PASS: Violation correctly detected (Lobotomy Prevention). Preservation: {preservation.mean().item()}")
    else:
        print("FAIL: Lobotomy not detected.")

def test_gray_zone():
    print("\nTesting Gray Zone Classification...")
    classifier = SelfReferenceAdmissibility.classify_gray_state
    
    # Binary State
    binary_state = torch.tensor([0.0, 1.0, 1.0])
    res = classifier(binary_state)
    print(f"State: {binary_state} -> {res}")
    
    # Gray State
    gray_state = torch.tensor([0.0, 0.5, 1.0])
    res = classifier(gray_state)
    print(f"State: {gray_state} -> {res}")
    
    if res == "Admissible Gray State":
        print("PASS: Gray Zone correctly identified.")
    else:
        print("FAIL: Gray Zone rejected.")

if __name__ == "__main__":
    test_implication_invariant()
    test_gray_zone()
