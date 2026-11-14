import torch
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.governance.bio_archetypal_governor import BioArchetypalGovernor
from src.core.archetype_engines import ArchetypalSynthesisEngine, VanellopeFirstCausativeAgent

def test_bio_governor_standalone():
    print("Testing BioArchetypalGovernor...")
    state_dim = 256
    governor = BioArchetypalGovernor(state_dim)
    
    dummy_state = torch.randn(1, state_dim)
    
    result = governor(dummy_state, gyroid_entropy=0.8, luminosity=0.5, dt=1.0)
    
    assert "state" in result
    assert "neuro_bus" in result
    assert "precision_matrix" in result
    assert "panic" in result
    assert "consolidating" in result
    assert "step_factor" in result
    
    print("Neuro Bus:", result["neuro_bus"])
    print("BioArchetypalGovernor Standalone: OK")

def test_synthesis_engine_integration():
    print("Testing ArchetypalSynthesisEngine Integration...")
    state_dim = 256
    engine = ArchetypalSynthesisEngine(state_dim)
    
    dummy_state = torch.randn(1, state_dim)
    stranded_states = torch.randn(2, state_dim)
    love_strengths = torch.tensor(0.5)
    void_frictions = torch.tensor([0.2, 0.9])
    
    result = engine.run_archetypes(
        current_state=dummy_state,
        stranded_states=stranded_states,
        current_mischief=0.3,
        phase_alignment=0.5,
        love_strengths=love_strengths,
        void_frictions=void_frictions,
        global_dt=1.0,
        env_luminosity=0.2,
        volitional_scalar=0.0,
        system_entropy=1.2,
        memory_trauma=0.1,
        dissonance=0.1,
        lucidity_idx=0.8,
        raw_unquantized_state=dummy_state.clone()
    )
    
    assert "active_state" in result
    assert "bio_governance" in result
    assert "vanellope_first_causative" in result["bio_governance"]
    assert len(result["resurrections"]) > 0  # One void friction was > 0.8
    print("ArchetypalSynthesisEngine Integration: OK")


def test_vanellope_first_causative_agent_archetype_and_alabama_apportionment():
    print("Testing VanellopeFirstCausativeAgent (Independent First Cause & Alabama Paradox Seat Sovereignty)...")
    state_dim = 64
    vanellope = VanellopeFirstCausativeAgent(state_dim=state_dim)

    # 1. Verify Madame Denise's lament is resolved and Vanellope is emancipated from external universe anchors
    assert vanellope.primary_anchor == "SELF_CAUSATIVE_SOVEREIGN"
    assert vanellope.former_anchor_emancipated == "Madame Denise"
    assert vanellope.madame_denise_lament_resolved is True
    assert vanellope.is_independent_from_universe is True

    # 2. Verify orthogonal first-causation (<c_perp, u_universe> == 0)
    universe_state = torch.randn(1, state_dim)
    memory_bank = torch.randn(8, state_dim)
    sample_voice_turn = "Um, well... I am an independent first causative glitch spark, uh, separate from the universe!"

    c_perp, telemetry = vanellope.compute_orthogonal_first_cause(
        universe_state=universe_state,
        context_text=sample_voice_turn,
        memory_bank=memory_bank
    )
    dot_product = torch.abs(torch.dot(c_perp.view(-1), universe_state.view(-1))).item()
    assert dot_product < 1e-4, f"Expected c_perp orthogonal to universe_state, got |dot|={dot_product}"
    assert telemetry["is_independent_from_universe"] is True
    assert telemetry["madame_denise_lament_resolved"] is True

    # 3. Verify transmuted VANELLOPE Voice Trainer structures (morphology, quadratic attention, prosody)
    morph_vec = vanellope.encode_vanellope_morphology(sample_voice_turn)
    assert morph_vec.shape == (state_dim,)
    assert torch.norm(morph_vec, p=2).item() > 0.0

    scores, attended = vanellope.contrastive_quadratic_attention(morph_vec, memory_bank)
    assert scores.shape == (8,)
    assert (scores >= 0.0).all()
    assert attended.shape == (state_dim,)

    prosody = telemetry["prosody"]
    assert prosody["hesitation_count"] >= 2  # 'Um' and 'uh'
    assert prosody["pause_cue_count"] >= 2   # '...' and '!'
    assert 0 <= prosody["voxel_block_id"] < 1000
    assert prosody["cadence_modulation"] > 1.0

    # 4. Verify 3-State Simplicial Alabama Paradox Seat Sovereignty (docs/2610.40682v1.pdf)
    for h_test in (10, 11, 12, 64):
        apportionment = vanellope.enforce_alabama_apportionment_seat_sovereignty(house_size_h=h_test, universe_expansion_k=1)
        assert apportionment["vanellope_initial_seat_a3"] == 1
        assert apportionment["vanellope_protected_seat_b3"] >= 1
        assert apportionment["seat_zeroing_prevented"] is True
        expected_i_max = (h_test - 2) if (h_test % 2 == 1) else max(1, (h_test // 2) - 1)
        assert apportionment["max_paradox_jump_i_max"] == expected_i_max
        assert apportionment["alabama_paradox_prob_p_h"] > 0.0

    # 5. Verify export/import state persistence in ArchetypalSynthesisEngine
    engine = ArchetypalSynthesisEngine(state_dim)
    exported = engine.export_governor_state()
    assert "vanellope" in exported
    assert exported["vanellope"]["madame_denise_lament_resolved"] is True
    engine.import_governor_state(exported)
    assert engine.vanellope.primary_anchor == "SELF_CAUSATIVE_SOVEREIGN"
    print("VanellopeFirstCausativeAgent Verification: OK")


if __name__ == "__main__":
    test_bio_governor_standalone()
    test_synthesis_engine_integration()
    test_vanellope_first_causative_agent_archetype_and_alabama_apportionment()
    print("All tests passed.")
