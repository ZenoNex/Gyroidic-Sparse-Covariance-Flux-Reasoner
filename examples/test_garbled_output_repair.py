"""
Test script for garbled output repair system.

Tests the Phase 1 repair components to fix the "nccmtsmneltcclrclcnl,tncsectsead" 
type garbled outputs caused by topological fractures.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from core.love_invariant_protector import LoveInvariantProtector, SoftSaturatedGates
from core.polynomial_coprime import PolynomialCoprimeConfig


def simulate_garbled_output():
    """Simulate the conditions that lead to garbled output."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    K = 5  # Number of functionals
    D = 4  # Polynomial degree + 1
    hidden_dim = 256
    
    print("🔧 Testing Garbled Output Repair System")
    print("=" * 50)
    
    # Simulate problematic residues (high consonant clustering)
    # These represent the "nccmtsmneltcclrclcnl" type patterns
    problematic_residues = torch.randn(batch_size, K, D, device=device)
    
    # Add spectral fragmentation (high-frequency dominance)
    high_freq_noise = torch.randn_like(problematic_residues) * 2.0
    problematic_residues += high_freq_noise
    
    # Add modulus drift (stale Bezout coefficients)
    drift_factor = torch.linspace(0.5, 2.0, K, device=device).unsqueeze(0).unsqueeze(-1)
    problematic_residues *= drift_factor
    
    # Simulate system state for Love protection
    system_state = torch.randn(batch_size, hidden_dim, device=device)
    
    return problematic_residues, system_state, device


def test_spectral_coherence_repair():
    """Test spectral coherence correction."""
    print("\n1. Testing Spectral Coherence Correction")
    print("-" * 40)
    
    problematic_residues, system_state, device = simulate_garbled_output()
    
    corrector = SpectralCoherenceCorrector(device=device)
    
    # Simulate garbled text (consonant clustering)
    garbled_text = "nccmtsmneltcclrclcnl,tncsectsead"
    
    print(f"Input text: {garbled_text}")
    print(f"Consonant clustering detected: {corrector.detect_consonant_clustering(garbled_text)}")
    
    # Apply correction
    corrected_signal = corrector.adaptive_coherence_correction(
        problematic_residues, garbled_text
    )
    
    diagnostics = corrector.get_diagnostics()
    print(f"Coherence threshold adjusted to: {diagnostics['theta_coherence']:.3f}")
    print(f"Soliton/Ergodic energy ratio: {diagnostics['energy_ratio']:.3f}")
    
    # Check if correction reduces high-frequency dominance
    original_std = problematic_residues.std()
    corrected_std = corrected_signal.std()
    print(f"Signal variance reduced: {original_std:.3f} → {corrected_std:.3f}")
    
    return corrected_signal


def test_bezout_refresh():
    """Test Bezout coefficient refresh."""
    print("\n2. Testing Bezout Coefficient Refresh")
    print("-" * 40)
    
    problematic_residues, _, device = simulate_garbled_output()
    K, D = problematic_residues.shape[1], problematic_residues.shape[2]
    
    bezout_refresh = BezoutCoefficientRefresh(K, D-1, device=device)
    
    # Simulate modulus drift
    print(f"Modulus drift detected: {bezout_refresh.detect_modulus_drift(problematic_residues)}")
    
    # Apply CRT correction
    corrected_residues = bezout_refresh.apply_crt_correction(problematic_residues)
    
    # Check if correction improves orthogonality
    original_corr = torch.corrcoef(problematic_residues.flatten(1).T).abs().mean()
    corrected_corr = torch.corrcoef(corrected_residues.flatten(1).T).abs().mean()
    
    print(f"Average correlation: {original_corr:.3f} → {corrected_corr:.3f}")
    print(f"Orthogonality improved: {corrected_corr < original_corr}")
    
    return corrected_residues


def test_chern_simons_gasket():
    """Test Chern-Simons gasket for logic leak repair."""
    print("\n3. Testing Chern-Simons Gasket")
    print("-" * 40)
    
    problematic_residues, _, device = simulate_garbled_output()
    K = problematic_residues.shape[1]
    
    gasket = ChernSimonsGasket(device=device)
    
    # Use proper polynomial co-prime functional system (anti-lobotomy)
    polynomial_config = PolynomialCoprimeConfig(
        k=K,
        degree=D - 1,
        basis_type='chebyshev',
        learnable=True,
        use_saturation=True,
        device=device
    )
    poly_coeffs = polynomial_config.get_coefficients_tensor()  # [K, D]
    
    # Check for logic leaks
    leak_detected = gasket.detect_logic_leak(problematic_residues)
    print(f"Logic leak detected: {leak_detected}")
    
    # Apply gasket repair
    repaired_residues = gasket.plug_logic_leak(problematic_residues, poly_coeffs)
    
    diagnostics = gasket.get_diagnostics()
    print(f"Twist energy: {diagnostics['twist_energy']:.6f}")
    print(f"Chern-Simons level: {diagnostics['level_k']}")
    
    # Check if repair reduces leak
    leak_after_repair = gasket.detect_logic_leak(repaired_residues)
    print(f"Logic leak after repair: {leak_after_repair}")
    
    return repaired_residues


def test_soliton_healing():
    """Test soliton stability healing."""
    print("\n4. Testing Soliton Stability Healing")
    print("-" * 40)
    
    problematic_residues, _, device = simulate_garbled_output()
    
    healer = SolitonStabilityHealer(device=device)
    
    # Simulate fractured soliton text (use the actual garbled pattern)
    fractured_text = "nccmtsmneltcclrclcnl"
    
    print(f"Fractured soliton detected: {healer.detect_fractured_soliton(fractured_text)}")
    
    # Apply healing
    healed_residues = healer.heal_fractured_soliton(problematic_residues, fractured_text)
    
    diagnostics = healer.get_diagnostics()
    print(f"Healing progress: {diagnostics['healing_progress']:.3f}")
    print(f"Alpha (ranging): {diagnostics['alpha']:.3f}")
    
    # Check if healing reduces fracture indicators
    original_variance = problematic_residues.var(dim=-1).mean()
    healed_variance = healed_residues.var(dim=-1).mean()
    print(f"Variance (fracture indicator): {original_variance:.3f} → {healed_variance:.3f}")
    
    return healed_residues


def test_love_protection():
    """Test Love invariant protection."""
    print("\n5. Testing Love Invariant Protection")
    print("-" * 40)
    
    _, system_state, device = simulate_garbled_output()
    hidden_dim = system_state.shape[1]
    
    protector = LoveInvariantProtector(love_dim=hidden_dim, device=device)
    
    # Get original Love vector
    original_love = protector.get_love_vector()
    print(f"Original Love vector norm: {torch.norm(original_love):.3f}")
    
    # Simulate ownership violation (system trying to optimize Love)
    protector.L += torch.randn_like(protector.L) * 0.1  # Simulate violation
    
    violation_detected = protector.detect_love_violation()
    print(f"Love violation detected: {violation_detected}")
    
    # Apply protection
    protected_love, diagnostics = protector.apply_love_protection(system_state)
    
    print(f"Protected Love vector norm: {torch.norm(protected_love):.3f}")
    print(f"Violation count: {diagnostics['violation_count']}")
    
    return protected_love


def test_soft_gates():
    """Test soft saturated gates."""
    print("\n6. Testing Soft Saturated Gates")
    print("-" * 40)
    
    problematic_residues, _, device = simulate_garbled_output()
    K, D = problematic_residues.shape[1], problematic_residues.shape[2]
    
    soft_gates = SoftSaturatedGates(K, D-1, device=device)
    
    # Apply soft saturation
    pas_h = 0.3  # Low phase alignment (incoherent regime)
    saturated_residues = soft_gates.apply_soft_saturation(problematic_residues, pas_h)
    
    diagnostics = soft_gates.get_diagnostics()
    print(f"System temperature (dt): {diagnostics['dt']:.3f}")
    print(f"Adaptive lambda: {diagnostics['lambda_adaptive']:.3f}")
    print(f"Fossilized functionals: {diagnostics['num_fossilized']}")
    
    # Check tri-state logic (should have zeros for silence)
    zero_ratio = (saturated_residues.abs() < 1e-6).float().mean()
    print(f"Silence ratio (tri-state): {zero_ratio:.3f}")
    
    return saturated_residues


def test_integrated_repair():
    """Test integrated repair pipeline."""
    print("\n7. Testing Integrated Repair Pipeline")
    print("=" * 50)
    
    problematic_residues, system_state, device = simulate_garbled_output()
    
    print("Applying full repair pipeline...")
    
    # 1. Spectral coherence correction
    corrector = SpectralCoherenceCorrector(device=device)
    signal_corrected = corrector.adaptive_coherence_correction(
        problematic_residues, "nccmtsmneltcclrclcnl"
    )
    
    # 2. Bezout refresh
    K, D = problematic_residues.shape[1], problematic_residues.shape[2]
    bezout_refresh = BezoutCoefficientRefresh(K, D-1, device=device)
    residues_corrected = bezout_refresh.apply_crt_correction(signal_corrected)
    
    # 3. Chern-Simons gasket
    gasket = ChernSimonsGasket(device=device)
    # Use proper polynomial co-prime functional system (anti-lobotomy)
    polynomial_config = PolynomialCoprimeConfig(
        k=K,
        degree=D - 1,
        basis_type='chebyshev',
        learnable=True,
        use_saturation=True,
        device=device
    )
    poly_coeffs = polynomial_config.get_coefficients_tensor()  # [K, D]
    residues_plugged = gasket.plug_logic_leak(residues_corrected, poly_coeffs)
    
    # 4. Soliton healing
    healer = SolitonStabilityHealer(device=device)
    residues_healed = healer.heal_fractured_soliton(residues_plugged, "nccmtsmnelt")
    
    # 5. Love protection
    protector = LoveInvariantProtector(love_dim=system_state.shape[1], device=device)
    protected_love, _ = protector.apply_love_protection(system_state)
    
    # 6. Soft gates
    soft_gates = SoftSaturatedGates(K, D-1, device=device)
    final_residues = soft_gates.apply_soft_saturation(residues_healed, 0.3)
    
    # Compare before and after
    original_chaos = problematic_residues.std()
    final_order = final_residues.std()
    
    print(f"\nRepair Results:")
    print(f"Original chaos (std): {original_chaos:.3f}")
    print(f"Final order (std): {final_order:.3f}")
    print(f"Chaos reduction: {((original_chaos - final_order) / original_chaos * 100):.1f}%")
    
    # Check for improved structure
    original_entropy = -torch.sum(torch.softmax(problematic_residues.flatten(), dim=0) * 
                                 torch.log_softmax(problematic_residues.flatten(), dim=0))
    final_entropy = -torch.sum(torch.softmax(final_residues.flatten(), dim=0) * 
                              torch.log_softmax(final_residues.flatten(), dim=0))
    
    print(f"Entropy: {original_entropy:.3f} → {final_entropy:.3f}")
    print(f"Structure improved: {final_entropy < original_entropy}")
    
    return final_residues


if __name__ == "__main__":
    print("🧪 Gyroidic Flux Reasoner - Garbled Output Repair Test")
    print("Testing fixes for 'nccmtsmneltcclrclcnl,tncsectsead' type outputs")
    print("=" * 60)
    
    try:
        # Test individual components
        test_spectral_coherence_repair()
        test_bezout_refresh()
        test_chern_simons_gasket()
        test_soliton_healing()
        test_love_protection()
        test_soft_gates()
        
        # Test integrated pipeline
        test_integrated_repair()
        
        print("\n✅ All repair tests completed successfully!")
        print("\nThe repair system should fix:")
        print("• Consonant clustering (spectral fragmentation)")
        print("• CRT modulus drift (stale Bezout coefficients)")
        print("• Logic leaks (topological holes)")
        print("• Fractured solitons (MC ruptures)")
        print("• Love vector scalarization")
        print("• Binary clipping (linguistic flow loss)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
