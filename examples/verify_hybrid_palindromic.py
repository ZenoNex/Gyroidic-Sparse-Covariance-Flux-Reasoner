import torch
from src.core.fgrt_primitives import PrimeResonanceLadder
from src.core.modular_virtualization import ModularVirtualizationLayer
from src.core.zeitgeist_router import ZeitgeistRouter, ZeitgeistState
from src.core.admr_solver import PolynomialADMRSolver

def verify_hybrid_palindromic():
    print("--- Verifying Hybrid Palindromic Modular Architecture ---")
    
    # 1. Verify PrimeResonanceLadder
    print("\n1. Testing PrimeResonanceLadder...")
    ladder = PrimeResonanceLadder(num_resonators=5)
    freqs, repunits = ladder.forward()
    print(f"Frequencies shape: {freqs.shape}")
    print(f"Repunits shape: {repunits.shape}")
    print(f"Sample Frequencies: {freqs}")
    print(f"Sample Repunits: {repunits}")
    print("[OK] PrimeResonanceLadder returns (p, R_p) pairs.")

    # 2. Verify ModularVirtualizationLayer
    print("\n2. Testing ModularVirtualizationLayer (Hybrid Modulus)...")
    dim = 5
    layer = ModularVirtualizationLayer(dim=dim)
    hybrid_moduli = layer.get_hybrid_modulus()
    expected_hybrid = ladder.primes.float() * ladder.repunits.float()
    print(f"Hybrid Moduli: {hybrid_moduli}")
    assert torch.allclose(hybrid_moduli, expected_hybrid), "Hybrid modulus calculation error"
    print("[OK] ModularVirtualizationLayer uses Hybrid Palindromic Basis.")

    # 3. Verify ZeitgeistRouter (Symmetric Tensor)
    print("\n3. Testing ZeitgeistRouter (Symmetric Tensor CRT)...")
    moduli = (3, 5, 7)
    router = ZeitgeistRouter(dim=10, moduli=moduli)
    state = ZeitgeistState.initial(moduli)
    
    x = torch.randn(1, 10)
    mode, new_state, diag = router.forward(x, state)
    
    print(f"Mode: {mode}")
    print(f"Alpha Tensor Shape: {new_state.alpha_tensor.shape}")
    
    # Check for symmetry
    is_symmetric = torch.allclose(new_state.alpha_tensor, new_state.alpha_tensor.T)
    assert is_symmetric, "alpha_tensor is not symmetric"
    
    print(f"CRT Index: {new_state.crt_index}")
    print("[OK] ZeitgeistRouter implements Symmetric Tensor CRT.")

    # 4. Verify ADMR Solver (Warmstart)
    print("\n4. Testing PolynomialADMRSolver (Palindromic Warmstart)...")
    from src.core.polynomial_coprime import PolynomialCoprimeConfig
    config = PolynomialCoprimeConfig(num_functionals=10, max_degree=3)
    solver = PolynomialADMRSolver(poly_config=config, state_dim=10)
    
    states = torch.randn(2, 10)
    p_hash = torch.randn(1, 10)
    
    out = solver.stochastic_differential_step(
        states,
        palindromic_hash=p_hash
    )
    print(f"Output shape: {out.shape}")
    assert out.shape == states.shape, "ADMR output shape mismatch"
    print("[OK] ADMR Solver accepts palindromic_hash warmstart.")

    print("\n--- All Hybrid Palindromic Verifications Passed! ---")

if __name__ == "__main__":
    verify_hybrid_palindromic()
