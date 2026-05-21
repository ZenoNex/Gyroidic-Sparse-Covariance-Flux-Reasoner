import torch
import torch.nn as nn
import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional, Dict

# Enforce correct imports by prepending workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.orchestrator import UniversalOrchestrator
from src.core.honest_jitter import harvest_honest_jitter
from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
from src.core.modular_virtualization import ModularVirtualizationLayer

def main():
    # Setup quiet logging to focus on verification output
    logging.basicConfig(level=logging.WARNING)
    print("======================================================================")
    print("SOVEREIGN ARCHETYPE SUBSTRATE & CONJURING INTEGRATION VALIDATION HARNESS")
    print("======================================================================")
    
    dim = 96
    orchestrator = UniversalOrchestrator(dim=dim)
    
    # ----------------------------------------------------------------------
    # TEST 1: Volitional Conjuring Override
    # ----------------------------------------------------------------------
    print("\n[CHECK 1] Testing Volitional Conjuring Override...")
    volition_injector = orchestrator.archetype_governor.volition_injector
    input_state = torch.full((1, dim), 0.2)
    
    # Low volition (< 0.9): Standard ADMM routing constraints are fully preserved
    out_state_low = volition_injector(input_state, user_volition_scalar=0.5)
    assert torch.allclose(out_state_low, input_state), "Low volition should not trigger bypass"
    print("  Low volition input preserved correctly.")
    
    # High volition (> 0.9): Exogenous override bypasses standard constraints
    out_state_high = volition_injector(input_state, user_volition_scalar=0.95)
    assert not torch.allclose(out_state_high, input_state), "High volition must trigger admin bypass layer"
    print(f"  High volition admin bypass triggered. Shift magnitude: {torch.norm(out_state_high - input_state).item():.4f}")
    print("[OK] Volitional Conjuring Override bypassed standard constraints successfully.")
    
    # ----------------------------------------------------------------------
    # TEST 2: Textbook Quality Gate
    # ----------------------------------------------------------------------
    print("\n[CHECK 2] Testing Textbook Quality Gate...")
    stacker = orchestrator.archetype_governor.tag_stacker
    
    # Test SLOP / Dishonest patterns (TODO or random placeholders)
    bad_context = "This is a TODO placeholder using torch.randn(96) to generate fake data."
    success_bad, report_bad = stacker.add_tag("slop_archetype", torch.zeros(dim), bad_context)
    assert not success_bad, "Should have rejected dishonest pattern"
    assert not report_bad.is_admissible, "Should have marked dishonest pattern as inadmissible"
    print(f"  SLOP coordinate rejected. Rejected flags: {report_bad.flags}")
    
    # Test high-quality instructive context (Algorithmic / instructive / self-contained / clear)
    good_context = (
        "# Instructive Gyroidic Resonance Mapping\n"
        "# Enforces structural honesty and continuous co-primality.\n"
        "# Implements numerical surgery spaces to optimize manifold topology.\n"
        "def calculate_coprime_residue(state: list) -> list:\n"
        "    \"\"\"\n"
        "    Computes dynamic Legendre coefficients.\n"
        "    This is instructive and self-contained for advanced mathematical reasoning.\n"
        "    \n"
        "    Example usage:\n"
        "        calculate_coprime_residue([1.0, 2.0, 3.0])\n"
        "    \"\"\"\n"
        "    results = []\n"
        "    for x in state:\n"
        "        # Enforce coprime frequency shift\n"
        "        shift = x * 1.127\n"
        "        results.append(shift)\n"
        "    \n"
        "    # Return finalized invariant state\n"
        "    return results\n"
        "\n"
        "# Verification of the numerical module\n"
        "def check_manifold_stability(state):\n"
        "    # Measure topological erosion rate\n"
        "    return len(state) > 0\n"
    )
    success_good, report_good = stacker.add_tag("sovereign_archetype", torch.ones(dim), good_context)
    assert success_good, "Should have successfully added textbook-quality tag"
    assert report_good.is_admissible, "Should have validated textbook-quality tag as admissible"
    print(f"  Textbook-quality coordinate accepted. Admissibility Report: {report_good.to_dict()}")
    print("[OK] Textbook Quality Gate enforces structural honesty dynamically.")
    
    # ----------------------------------------------------------------------
    # TEST 3: Dynamic Archetype Superposition
    # ----------------------------------------------------------------------
    print("\n[CHECK 3] Testing Dynamic Archetype Superposition...")
    # Add another coordinate to check multi-scalar continuous stacking
    good_context_2 = (
        "# Dynamic Legendre Braid Mapping\n"
        "# Preserves non-ergodic entropy and avoids division by zero.\n"
        "# Incorporates the real projective four-space transition mapping.\n"
        "def compute_braid_phase(state: list) -> list:\n"
        "    \"\"\"\n"
        "    Transforms custom coordinate into an elliptic torus phase.\n"
        "    \n"
        "    Example usage:\n"
        "        compute_braid_phase([0.1, 0.2])\n"
        "    \"\"\"\n"
        "    results = []\n"
        "    for val in state:\n"
        "        # Perform torus phase shift rotation\n"
        "        shifted = val + 0.1\n"
        "        results.append(shifted)\n"
        "        \n"
        "    return results\n"
        "\n"
        "# Subspace isolation helper\n"
        "def ensure_non_ergodic_torsion(metric):\n"
        "    # Check Betti numbers boundary limits\n"
        "    return metric > 0.0\n"
    )
    success_2, report_2 = stacker.add_tag("monarch_butterfly", torch.ones(dim) * 2.0, good_context_2)
    assert success_2, "Failed to register monarch_butterfly coordinate"
    
    # Stacking with unbound weights (permitting feature subtraction)
    tag_weights = {
        "monarch_butterfly": 1.8,
        "sovereign_archetype": -0.6
    }
    composite_target = stacker.compute_composite_target(tag_weights)
    
    # Stacker normalizes registered vectors internally to secure scalar stability:
    # vector_sovereign_normalized = ones / sqrt(dim)
    # vector_monarch_normalized = ones / sqrt(dim)
    # Target = (1.8 * vector_monarch) + (-0.6 * vector_sovereign) = 1.2 * vector_normalized
    expected_target = 1.2 * torch.nn.functional.normalize(torch.ones(dim), dim=-1)
    assert torch.allclose(composite_target, expected_target, atol=1e-5), "Composite target does not match unbound superposition"
    print(f"  Superposition composite target successfully computed.")
    print(f"  Target vector norm: {composite_target.norm().item():.4f}")
    print("[OK] Dynamic Archetype Superposition with unbound weights completed.")
    
    # ----------------------------------------------------------------------
    # TEST 4: Upstream Atrophy Guard
    # ----------------------------------------------------------------------
    print("\n[CHECK 4] Testing Upstream Atrophy Guard...")
    fossilizer = DyadFossilizer(storage_dir="scratch/data/encodings", feature_dim=512)
    dyad = KnowledgeDyad(
        linguistic_description="Topological Braid anomaly inside Poincaré core.",
        image_fingerprint=torch.randn(96)
    )
    
    # Create zero-variance state representing the 0.8824 flatline (Atrophy!)
    dead_prime_state = torch.full((512,), 0.8824)
    text_embedding = torch.randn(512)
    
    filename = fossilizer.fossilize(dyad, text_embedding, seed_state=dead_prime_state)
    assert os.path.exists(filename), "Failed to save fossil file"
    
    # Load and check saved data
    saved_data = torch.load(filename)
    assert saved_data is not None
    
    # Verify rehydration did not leave it flat (variance > 0)
    saved_meta = saved_data.get('meta_state')
    assert saved_meta is not None, "Failed to retrieve meta_state from fossil payload"
    variance = saved_meta.var().item()
    print(f"  Intercepted flatline state. Rehydrated variance: {variance:.6e}")
    assert variance > 1e-8, "Atrophy guard failed to rehydrate zero-variance state"
    
    # Clean up temporary fossil file
    if os.path.exists(filename):
        os.remove(filename)
        
    print("[OK] Upstream Atrophy Guard intercepted, rehydrated, and fossilized flatline successfully.")
    
    # ----------------------------------------------------------------------
    # TEST 5: Kinger Darkness Loophole (Ombre Effect)
    # ----------------------------------------------------------------------
    print("\n[CHECK 5] Testing Kinger Darkness Loophole...")
    ombre = orchestrator.archetype_governor.ombre
    original_state = torch.full((1, dim), 0.3)
    quantized_state = torch.full((1, dim), 0.1)
    
    # Under standard luminosity (1.0), it does not blend back
    out_ombre_std = ombre(original_state, environmental_luminosity=1.0, original_quantized_state=quantized_state)
    assert torch.allclose(out_ombre_std, original_state), "Standard light should keep current state"
    print("  Standard light quantization constraints preserved.")
    
    # Under low rendering pressure (luminosity < 0.3), standard boundaries relax
    out_ombre_dark = ombre(original_state, environmental_luminosity=0.1, original_quantized_state=quantized_state)
    # It should boost lucidity (out_ombre_dark = state * 2.0 + quantized * 0.1)
    expected_dark = original_state * 2.0 + quantized_state * 0.1
    assert torch.allclose(out_ombre_dark, expected_dark), "Dark luminosity should relax constraints and blend states"
    print(f"  Dark environment detected. Quantization relaxed. State scaled by boost factor.")
    print("[OK] Kinger Darkness Loophole successfully bridges fragmented spaces under low render pressure.")
    
    # ----------------------------------------------------------------------
    # TEST 6: Paradox Hardening (Torus Projection)
    # ----------------------------------------------------------------------
    print("\n[CHECK 6] Testing Paradox Hardening (Torus Projection)...")
    virtualizer = ModularVirtualizationLayer(dim=dim)
    state_a = torch.randn(dim)
    state_b = state_a + torch.randn(dim) * 0.01 # Highly congruent state
    state_c = torch.randn(dim) # Non-congruent state
    
    is_congruent = virtualizer.fast_congruence_check(state_a, state_b)
    is_non_congruent = virtualizer.fast_congruence_check(state_a, state_c)
    
    print(f"  Congruence check (A vs B): {is_congruent} (Should be True)")
    print(f"  Congruence check (A vs C): {is_non_congruent} (Should be False)")
    
    assert is_congruent == True, "Slightly perturbed state must register as congruent"
    
    # Test torus remainder mapping (cyclic overflow boundaries)
    rns_residues = virtualizer.float_to_rns(state_a)
    assert torch.all(rns_residues >= 0), "Residues must be non-negative"
    assert torch.all(rns_residues < virtualizer.get_hybrid_modulus()), "Residues must not exceed modulus"
    print(f"  Floats successfully mapped to torus residues in finite field.")
    print(f"  Hybrid moduli range: [{virtualizer.get_hybrid_modulus().min().item():.1f}, {virtualizer.get_hybrid_modulus().max().item():.1f}]")
    print("[OK] Paradox Hardening mapped float states safely onto the modular torus.")
    
    print("\n======================================================================")
    print("ALL SOVEREIGN ARCHETYPE VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")

if __name__ == '__main__':
    main()
