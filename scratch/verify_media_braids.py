import torch
import torch.nn.functional as F
from src.ui.diegetic_backend import DiegeticPhysicsEngine
import os

def test_media_braid_logic():
    print("⬡ TESTING HETEROGENEOUS MEDIA BRAID...")
    
    # Initialize engine
    # Note: Using a minimal setup if possible, or mocking necessary parts
    engine = DiegeticPhysicsEngine(device='cpu')
    
    # Mock media residues (Chebyshev coefficients)
    # Image Residue (K=32, 3 channels = 96 coeffs)
    img_data = {
        'L': [0.1] * 32,
        'Cr': [0.2] * 32,
        'Cb': [0.3] * 32
    }
    
    # Audio Residue (K=26 harmonics)
    audio_data = {
        'chebyshev_harmonics': [0.5] * 26
    }
    
    # Test 1: Image -> Audio
    print("\n--- TEST 1: Image -> Audio ---")
    engine.meta_state = torch.zeros((1, engine.dim))
    chain_1 = [
        {'type': 'image', 'data': img_data},
        {'type': 'audio', 'data': audio_data}
    ]
    engine.process_input("TEST_BRAID_1", media_chain=chain_1, commutativity='media_first', generate_response=False)
    state_1 = engine.meta_state.clone()
    print(f"State 1 mean: {state_1.mean().item():.6f}")
    
    # Test 2: Audio -> Image
    print("\n--- TEST 2: Audio -> Image ---")
    engine.meta_state = torch.zeros((1, engine.dim))
    chain_2 = [
        {'type': 'audio', 'data': audio_data},
        {'type': 'image', 'data': img_data}
    ]
    engine.process_input("TEST_BRAID_2", media_chain=chain_2, commutativity='media_first', generate_response=False)
    state_2 = engine.meta_state.clone()
    print(f"State 2 mean: {state_2.mean().item():.6f}")
    
    # Test 3: Symmetric (Image + Audio)
    print("\n--- TEST 3: Symmetric (Simultaneous) ---")
    engine.meta_state = torch.zeros((1, engine.dim))
    engine.process_input("TEST_BRAID_3", media_chain=chain_1, commutativity='symmetric', generate_response=False)
    state_sym = engine.meta_state.clone()
    print(f"State Symmetric mean: {state_sym.mean().item():.6f}")
    
    # Verification: Non-Commutativity
    diff = torch.norm(state_1 - state_2).item()
    print(f"\n[VERIFICATION] Path Difference ||State1 - State2||: {diff:.8f}")
    
    if diff > 1e-6:
        print("✅ SUCCESS: Systems are NON-COMMUTATIVE. Path matters.")
    else:
        print("❌ FAILURE: Systems are commutative. Ordering failed to bias manifold.")

    # Verification: Braid vs Symmetric
    diff_sym = torch.norm(state_1 - state_sym).item()
    print(f"[VERIFICATION] Sequential vs Symmetric difference: {diff_sym:.8f}")

if __name__ == "__main__":
    test_media_braid_logic()
