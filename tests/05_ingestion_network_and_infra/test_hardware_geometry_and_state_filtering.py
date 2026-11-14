import pytest
import torch
from src.core.modular_virtualization import ModularVirtualizationLayer
from src.core.fgrt_primitives import CALMCollapseDetector
from src.topology.alexander_wild_embedding import AlexanderWildEmbeddingOperator, AlexanderTransversalityReport
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
from src.core.honest_jitter import harvest_honest_jitter
from src.core.pressure_typing import StructuralPressure
from src.training.gdpo_trainer import GDPOSovereigntyPressureComputer
from src.data.gltf_splat_ingestor import GltfSplatIngestionPipeline


def test_bb5_collatz_state_machine_parity_filtering():
    """Test 1: BB(5) Collatz Tree Coral and O(1) Repunit-CRT parity filtering."""
    layer = ModularVirtualizationLayer(dim=8)

    # Candidate with same LSB parity as target -> True
    cand_even = torch.tensor([4, 6, 8, 10])
    target_even = torch.tensor([2, 12, 14, 16])
    assert torch.all(layer.repunit_crt_sparse_probe(cand_even, target_even))

    cand_odd = torch.tensor([5, 7, 9, 11])
    target_odd = torch.tensor([1, 3, 13, 15])
    assert torch.all(layer.repunit_crt_sparse_probe(cand_odd, target_odd))

    # Opposing parity -> False (O(1) fast rejection)
    assert not torch.any(layer.repunit_crt_sparse_probe(cand_even, target_odd))
    assert not torch.any(layer.repunit_crt_sparse_probe(cand_odd, target_even))


def test_calm_collatz_limit_cycle_debt_veto():
    """Test 1b: CALMCollapseDetector monitors cycle debt and triggers veto on Collatz limit traps."""
    detector = CALMCollapseDetector(warning_threshold=0.8, cycle_debt_threshold=0.5)
    history = torch.randn(2, 10, 8)

    # No cycle debt, low correlation
    is_col, corr = detector(history, cycle_debt=0.1)
    assert not is_col.any()

    # High cycle debt (> 0.5) triggers entropic veto
    is_col_veto, _ = detector(history, cycle_debt=0.75)
    assert is_col_veto.all()


def test_alexander_wild_embedding_rp4_transversality():
    """Test 2: Alexander 1924 Wild Embedding Operator in RP^4 with Bouligand tangent cone."""
    operator = AlexanderWildEmbeddingOperator(dim=16, num_cubes=5)
    state = torch.randn(2, 16)

    report = operator(state)
    assert isinstance(report, AlexanderTransversalityReport)
    assert report.is_transversal in (True, False)
    assert report.shear_flux >= 0.0
    assert report.bouligand_kink_intensity >= 0.0
    assert report.reconstructed_state.shape == (2, 16)


def test_hardware_sovereignty_warp_and_jitter():
    """Test 3: Hardware sovereignty 64 work-item SRAM fusion configuration and honest jitter."""
    # Workgroup size 64 = exactly 2 warps on Pascal GP107
    engine = SiliconSovereigntyEngine(use_gpu=False)
    assert engine._shared_max_work_group_size >= 64

    # Honest jitter entropy harvesting
    jitter = harvest_honest_jitter((4, 8), scaled=True)
    assert jitter.shape == (4, 8)
    assert not torch.isnan(jitter).any()


def test_gdpo_decoupled_normalization_and_anti_scalarization():
    """Test 4: GDPO decoupled normalization and StructuralPressure type-level anti-scalarization."""
    # 1. StructuralPressure raises on cross-domain scalarization
    p_silicon = StructuralPressure(torch.tensor(0.4), domain="silicon")
    p_thermal = StructuralPressure(torch.tensor(0.6), domain="thermal")
    p_silicon_b = StructuralPressure(torch.tensor(0.2), domain="silicon")

    # Same domain sum is permitted
    sum_silicon = p_silicon + p_silicon_b
    assert sum_silicon.domain == "silicon"
    assert torch.isclose(sum_silicon.value, torch.tensor(0.6))

    # Cross domain sum raises ValueError
    with pytest.raises(ValueError, match="Scalarization Trap"):
        _ = p_silicon + p_thermal

    # 2. GDPOSovereigntyPressureComputer decoupled normalization
    computer = GDPOSovereigntyPressureComputer(num_pressures=4)
    pressures = torch.rand(2, 5, 4)
    values = torch.rand(2, 5)
    dones = torch.zeros(2, 5)
    weights = torch.ones(4) / 4.0

    advantages, diag = computer.compute_sovereignty_pressures(pressures, values, dones, weights)
    assert advantages.shape == (2, 5, 4)
    assert "decoupled_pressures" in diag


def test_dmca_safe_splat_betti_extraction():
    """Test 5: DMCA-safe 3D Splat topological Betti invariant and density field extraction."""
    pipeline = GltfSplatIngestionPipeline(target_dim=32)

    # Synthetic splat file ingestion
    fake_glb_path = "non_existent_model.glb"
    betti_info = pipeline.extract_topological_betti_invariants(fake_glb_path)

    assert betti_info["beta_0"] >= 1
    assert betti_info["beta_1"] >= 0
    assert betti_info["density_field"].shape == (32,)
    assert betti_info["raw_mesh_retained"] is False # DMCA/IP safe guarantee
