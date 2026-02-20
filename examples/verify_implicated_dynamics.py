import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybrid_backend import HybridAI

def test_implicated_dynamics():
    print("--- Verifying Implicated System Dynamics (Δ) ---")
    ai = HybridAI(use_spectral_correction=True)
    
    # 1. Test Damage Accumulation (Δ)
    print("\n[1] Testing Damage Accumulation...")
    # Simulate a series of messages that might cause stress/rupture
    messages = [
        "What is the identity of the modular remainder?",
        "Is the remainder also the whole?",
        "The remainder is not the whole, but the whole is the remainder.",
        "p and not p are both true in the Birkhoff polytope."
    ]
    
    for msg in messages:
        res = ai.process_text(msg)
        damage = res['diagnostics'].get('damage_delta', 0.0)
        print(f"Input: '{msg}' -> Response: '{res['response'][:50]}...' -> Delta: {damage:.4f}")

    final_damage = float(ai.damage_residue.norm())
    print(f"Final Damage Norm: {final_damage:.4f}")
    
    # 2. Test Order-Sensitivity (Non-Commutativity)
    print("\n[2] Testing Order-Sensitivity (L_a L_b != L_b L_a)...")
    
    # Sequence A then B
    ai_a = HybridAI(use_spectral_correction=True)
    ai_a.process_text("A")
    ai_a.process_text("B")
    state_ab = ai_a.damage_residue.clone()
    
    # Sequence B then A
    ai_b = HybridAI(use_spectral_correction=True)
    ai_b.process_text("B")
    ai_b.process_text("A")
    state_ba = ai_b.damage_residue.clone()
    
    diff = torch.norm(state_ab - state_ba).item()
    print(f"Commutativity Difference ||S_ab - S_ba||: {diff:.6f}")
    if diff > 1e-6:
        print("✅ Order-Sensitivity Verified (Laryngeal Operator is non-commutative)")
    else:
        print("❌ Commutativity detected - Check Laryngeal Operator logic")

    # 3. Test Response Degradation
    print("\n[3] Testing Response Degradation...")
    # Manually spike damage to see degradation
    ai.damage_residue = torch.randn(256) * 5.0
    res = ai.process_text("Status check.")
    print(f"High Damage Response: {res['response']}")
    
    if any(c in res['response'] for c in ['Δ', '⊥', '†', '◊', '∑', '∏']):
        print("✅ Response Degradation Verified (Paraconsistent Glitching active)")
    else:
        print("❌ No degradation detected at high damage level")

if __name__ == "__main__":
    test_implicated_dynamics()
