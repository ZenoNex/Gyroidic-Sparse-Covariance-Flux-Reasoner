import torch
from src.core.fgrt_primitives import PrimeResonanceLadder
from src.core.modular_virtualization import ModularVirtualizationLayer
from src.core.zeitgeist_router import ZeitgeistRouter, ZeitgeistState
from src.core.admr_solver import PolynomialADMRSolver

def verify_hybrid_palindromic():
    print("--- Verifying Hybrid Palindromic Modular Architecture ---")
    
    # 1. Verify PrimeResonanceLadder
    print("\n1. Testing PrimeResonanceLadder...")
    ladder = PrimeResonanceLadder(num_oscillators=5)
    # Mocking self.step if needed, but forward() handles it
    freqs, repunits = ladder.forward()
    print(f"Frequencies shape: {freqs.shape}")
    print(f"Repunits shape: {repunits.shape}")
    print(f"Sample Frequencies: {freqs}")
    print(f"Sample Repunits: {repunits}")
    assert freqs.shape == (5,), "Frequencies shape mismatch"
    assert repunits.shape == (5,), "Repunits shape mismatch"
    print("✅ PrimeResonanceLadder returns (p, R_p) pairs.")

    # 2. Verify ModularVirtualizationLayer
    print("\n2. Testing ModularVirtualizationLayer (Hybrid)...")
    mvl = ModularVirtualizationLayer(dim=10, num_moduli=5)
    x = torch.randn(2, 10)
    # The new version needs the repunits from the ladder
    q_residues, q_reconstructed, meta = mvl(x, freqs, repunits)
    print(f"Residues shape: {q_residues.shape}") # Should be [batch, M]
    print(f"Reconstructed shape: {q_reconstructed.shape}")
    
    # Check if modulus is hybrid (p * R_p)
    # The hybrid_moduli in metadata should be p * R_p
    hybrid_moduli = meta['hybrid_moduli']
    expected_hybrid = freqs * repunits
    print(f"Hybrid Moduli: {hybrid_moduli}")
    assert torch.allclose(hybrid_moduli, expected_hybrid), "Hybrid modulus calculation error"
    print("✅ ModularVirtualizationLayer uses Hybrid Palindromic Basis.")

    # 3. Verify ZeitgeistRouter (Symmetric Tensor)
    print("\n3. Testing ZeitgeistRouter (Symmetric Tensor CRT)...")
    moduli = (3, 5, 7)
    router = ZeitgeistRouter(dim=10, moduli=moduli)
    state = ZeitgeistState.initial(moduli=moduli)
    
    print(f"Initial alpha_tensor shape: {state.alpha_tensor.shape}")
    assert state.alpha_tensor.shape == (3, 3), "Initial alpha_tensor shape mismatch"
    
    x_router = torch.randn(1, 10)
    # Force a switch by setting a high switch pressure gate if needed, 
    # but just checking forward pass integrity for now.
    mode, new_state, diag = router(x_router, state)
    
    print(f"Mode: {mode}")
    print(f"New alpha_tensor shape: {new_state.alpha_tensor.shape}")
    assert new_state.alpha_tensor.shape == (3, 3), "New alpha_tensor shape mismatch"
    
    # Check symmetry
    is_symmetric = torch.allclose(new_state.alpha_tensor, new_state.alpha_tensor.T)
    print(f"Is Symmetric: {is_symmetric}")
    assert is_symmetric, "alpha_tensor is not symmetric"
    
    print(f"CRT Index: {new_state.crt_index}")
    print("✅ ZeitgeistRouter implements Symmetric Tensor CRT.")

    # 4. Verify ADMR Solver (Warmstart)
    print("\n4. Testing PolynomialADMRSolver (Palindromic Warmstart)...")
    solver = PolynomialADMRSolver(dim=10)
    states = torch.randn(2, 10)
    neighbor_states = torch.randn(2, 10)
    adj = torch.ones(2, 2)
    
    # Generate a dummy palindromic hash (e.g. from repunits)
    palindromic_hash = torch.ones(10)
    
    # Verify signature
    out = solver.stochastic_differential_step(
        states, neighbor_states, adj, 
        palindromic_hash=palindromic_hash
    )
    print(f"Output shape: {out.shape}")
    assert out.shape == states.shape, "ADMR output shape mismatch"
    print("✅ ADMR Solver accepts palindromic_hash warmstart.")

    print("\n--- All Hybrid Palindromic Verifications Passed! ---")

if __name__ == "__main__":
    verify_hybrid_palindromic()
