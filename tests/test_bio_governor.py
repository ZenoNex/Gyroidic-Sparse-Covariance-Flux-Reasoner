import torch
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.governance.bio_archetypal_governor import BioArchetypalGovernor
from src.core.archetype_engines import ArchetypalSynthesisEngine

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
    assert len(result["resurrections"]) > 0  # One void friction was > 0.8
    print("ArchetypalSynthesisEngine Integration: OK")

if __name__ == "__main__":
    test_bio_governor_standalone()
    test_synthesis_engine_integration()
    print("All tests passed.")
