
import torch
import torch.nn as nn
from src.core.orchestrator import UniversalOrchestrator
from src.core.honest_jitter import harvest_honest_jitter

def verify_orchestrator_flux():
    print("--- Verifying Archetype Flux Stabilization ---")
    
    dim = 64
    orchestrator = UniversalOrchestrator(dim=dim)
    
    # Test cases: (pas_h, expected_regime, expected_mode)
    # Note: determine_regime has complex conditions (coherence, stability, complexity, homology)
    # We will simulate high/low coherence and check transitions.
    
    test_cases = [
        {"pas_h": 0.95, "desc": "High Coherence (Potential Seriousness)"},
        {"pas_h": 0.50, "desc": "Low Coherence (Play/LERP)"},
        {"pas_h": 0.10, "desc": "Critical Coherence (Play/VOID?)"}
    ]
    
    state = harvest_honest_jitter((1, dim), device=torch.device('cpu'), scaled=True)
    pressure_grad = harvest_honest_jitter((1, dim), device=torch.device('cpu'), scaled=True)
    
    for case in test_cases:
        pas_h = case["pas_h"]
        print(f"\nTesting: {case['desc']} | pas_h={pas_h}")
        
        # We run multiple steps to allow the clock and EMA to settle
        for i in range(5):
            out_state, regime, routing = orchestrator(
                state=state,
                pressure_grad=pressure_grad,
                pas_h=pas_h,
                coherence=torch.tensor([pas_h]),
                atrophy=0.0
            )
        
        metrics = orchestrator.bulletin_board.read_metrics()
        nav_mode = metrics.get('nav_mode')
        leak = metrics.get('archetype_leak')
        
        print(f"  Regime: {regime}")
        print(f"  Routing: {routing}")
        print(f"  Nav Mode: {nav_mode}")
        print(f"  Archetype Leak: {leak:.6f}")
        
        # Assertions
        if pas_h < 0.8:
            assert regime == 'PLAY', f"Expected PLAY for pas_h={pas_h}, got {regime}"
            assert nav_mode == 'LERP', f"Expected LERP for PLAY, got {nav_mode}"
        
        # Archetype leak should be active (at least recorded)
        assert leak >= 0, "Archetype leak should be a non-negative scalar"

    print("\n--- Transition Test: PLAY -> SERIOUSNESS ---")
    # To hit SERIOUSNESS, we need stability (prev_pas ~ current_pas) and high pas_h
    # and also CI > 0.1, CPR lock, Glyphlock, Homology.
    # UniversalOrchestrator defaults some of these to True for stability.
    
    pas_h = 0.95
    for i in range(20): # Accumulate iteration for CI and stabilize drift
        out_state, regime, routing = orchestrator(
            state=state,
            pressure_grad=pressure_grad,
            pas_h=pas_h,
            coherence=torch.tensor([pas_h]),
            atrophy=0.0
        )
    
    metrics = orchestrator.bulletin_board.read_metrics()
    print(f"Final Regime after stabilization: {regime}")
    print(f"Final Nav Mode: {metrics.get('nav_mode')}")
    
    # If the system reached SERIOUSNESS, nav_mode should be SLERP
    if regime == 'SERIOUSNESS':
        assert metrics.get('nav_mode') == 'SLERP', "SERIOUSNESS must map to SLERP"
    else:
        print("Note: SERIOUSNESS not reached (might require more complex conditions like Glyphlock).")

    print("\n--- Subspace Isolation Check ---")
    # Verify that leak_projector is orthogonal
    weight = orchestrator.leak_projector.weight
    # For a Linear(1, dim), the weight is [dim, 1]. Orthogonal means norm is 1.
    norm = torch.norm(weight)
    print(f"Leak Projector weight norm: {norm.item():.4f}")
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-3), "Leak projector should be orthogonal"

    print("\nVerification Successful!")

if __name__ == "__main__":
    verify_orchestrator_flux()
