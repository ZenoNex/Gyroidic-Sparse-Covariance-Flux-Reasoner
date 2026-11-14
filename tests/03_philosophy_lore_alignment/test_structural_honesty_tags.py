import os
import json
import tempfile
import torch
import pytest

os.environ["Sovereign_Disable_Background"] = "1"

from src.core.agent_substrate_bridge import AgentSubstrateBridge
from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer
from src.topology.embedding_graph import KnowledgeFossilNode, GyroidicGraphManager
from src.ui.voxelboxter_simulation import (
    Block,
    StructuralGraph,
    FloraComponent,
    VoxelboxterEngine
)


def test_agent_substrate_bridge_public_domain_tag_preservation():
    """Verify that AgentSubstrateBridge preserves public domain tags under 17 U.S.C. 102(b)."""
    bridge = AgentSubstrateBridge(device="cpu")

    # Test individual tag admissibility
    assert bridge.is_public_domain_or_structural_tag("crt_1_2_3")
    assert bridge.is_public_domain_or_structural_tag("b0_100")
    assert bridge.is_public_domain_or_structural_tag("agent_smith")
    assert bridge.is_public_domain_or_structural_tag("math.dg")
    assert bridge.is_public_domain_or_structural_tag("quantum_gravity")
    assert bridge.is_public_domain_or_structural_tag("pirangi_cashew")
    assert bridge.is_public_domain_or_structural_tag("uberon:0001234")
    assert bridge.is_public_domain_or_structural_tag("public_domain")
    assert bridge.is_public_domain_or_structural_tag("17_usc_102b")

    # Test full sanitization pipeline
    payload = {
        "description": "A long copyrighted expressive narrative that should be hashed",
        "tags": [
            "crt_3_5_7",
            "math.dg",
            "differential_geometry",
            "anacardium_occidentale",
            "public_domain",
            "This is a forbidden long expressive narrative sentence with spaces and punctuation!"
        ],
        "pot_identity_crt": (3, 5, 7),
        "blake2s_digest": "abcdef0123456789",
        "meta_state_shielded": torch.randn(64),
        "raw_text": "verbatim text to remove"
    }

    sanitized = bridge.sanitize_and_verify_non_memorization(payload)

    # Verbatim text removed
    assert "raw_text" not in sanitized

    # Description hashed/replaced with non-invertible ID
    assert sanitized["description"].startswith("structural_soliton_crt_3_5_7")
    assert "prompt_sha256" in sanitized

    # Public domain and structural tags preserved
    tags = sanitized["tags"]
    assert "crt_3_5_7" in tags
    assert "agent_smith" in tags
    assert "structural_soliton" in tags
    assert "math.dg" in tags
    assert "differential_geometry" in tags
    assert "anacardium_occidentale" in tags
    assert "public_domain" in tags
    assert "This is a forbidden long expressive narrative sentence with spaces and punctuation!" not in tags

    pd_tags = sanitized["public_domain_tags"]
    assert "math.dg" in pd_tags
    assert "differential_geometry" in pd_tags
    assert "anacardium_occidentale" in pd_tags


def test_dyad_fossilizer_public_domain_tag_retention():
    """Verify that DyadFossilizer retains public domain metadata tags upon serialization."""
    temp_dir = tempfile.mkdtemp()
    try:
        fossilizer = DyadFossilizer(storage_dir=temp_dir, feature_dim=32)

        dyad = KnowledgeDyad(
            linguistic_description="Riemannian Ricci curvature flow on non-commutative surfaces",
            metadata={
                "source": "arxiv",
                "arxiv_id": "2401.99999",
                "categories": ["math.DG", "math.AP"],
                "public_domain_tags": ["ricci_flow", "riemannian_geometry", "public_domain"],
                "tags": ["ricci_flow", "riemannian_geometry", "public_domain"]
            }
        )

        dummy_text_emb = torch.randn(32)
        filepath = fossilizer.fossilize(dyad, text_embedding=dummy_text_emb)

        assert os.path.exists(filepath)
        saved_data = torch.load(filepath, map_location="cpu")

        # Verify saved payload has tags and public_domain_tags
        assert "tags" in saved_data
        assert "public_domain_tags" in saved_data
        assert "ricci_flow" in saved_data["tags"]
        assert "riemannian_geometry" in saved_data["tags"]
        assert any(t.startswith("crt_") for t in saved_data["tags"])
        assert "ricci_flow" in saved_data["public_domain_tags"]
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_graph_manager_public_domain_tag_export():
    """Verify that KnowledgeFossilNode and GyroidicGraphManager export public domain tags."""
    metrics = {
        "tags": ["crt_2_3_5", "b0_100", "quantum_gravity", "birkhoff_cone"],
        "public_domain_tags": ["quantum_gravity", "birkhoff_cone"],
        "tag_weights": {"quantum_gravity": 1.2, "birkhoff_cone": 0.8},
        "chiral_score": 0.45,
        "spectral_entropy": 0.88,
        "type": "fossil"
    }

    node = KnowledgeFossilNode(
        node_id="fossil_test_01",
        state=torch.randn(32),
        text="Quantum gravity test node",
        metrics=metrics
    )

    assert "quantum_gravity" in node.tags
    assert "quantum_gravity" in node.public_domain_tags
    assert "crt_2_3_5" not in node.public_domain_tags

    manager = GyroidicGraphManager(data_dir=tempfile.gettempdir(), dim=32)
    manager.nodes.append(node)

    # Test export_graph_json
    graph_json = json.loads(manager.export_graph_json())
    assert len(graph_json["nodes"]) == 1
    n_data = graph_json["nodes"][0]
    assert "public_domain_tags" in n_data
    assert "quantum_gravity" in n_data["public_domain_tags"]
    assert "tags" in n_data
    assert "quantum_gravity" in n_data["tags"]

    # Test query_procedural_graph (within 450.0 throttled render distance)
    proc_graph = manager.query_procedural_graph(
        camera_pos=(0.0, 0.0, 200.0),
        gaze_dir=(0.0, 0.0, -1.0),
        fov_deg=90.0,
        render_distance=1000.0
    )
    assert len(proc_graph["nodes"]) == 1
    pn_data = proc_graph["nodes"][0]
    assert "public_domain_tags" in pn_data
    assert "quantum_gravity" in pn_data["public_domain_tags"]


def test_voxelboxter_structural_honesty_and_public_domain_tags():
    """Verify that Voxelboxter blocks, constructs, and engine aggregate public domain tags."""
    engine = VoxelboxterEngine(device="cpu")

    # 1. Test Block tags
    block = Block(
        local_cell=(0, 0, 0),
        rotation=(0, 0, 0, 1),
        material_id=1,
        health=100.0,
        tags=["construct_alpha", "b_spline_spine"],
        public_domain_tags=["construct_alpha", "b_spline_spine", "public_domain_geometry"]
    )
    assert "b_spline_spine" in block.tags
    assert "public_domain_geometry" in block.public_domain_tags

    # 2. Test StructuralGraph tag aggregation
    graph = StructuralGraph()
    graph.add_block(block)
    assert "b_spline_spine" in graph.get_tags()
    assert "public_domain_geometry" in graph.get_public_domain_tags()

    construct_dict = graph.to_dict()
    assert construct_dict["total_blocks"] == 1
    assert "public_domain_geometry" in construct_dict["public_domain_tags"]

    # 3. Test FloraComponent botanical public domain tags
    flora = FloraComponent(entity_id="flora_test_tree")
    assert "anacardium_occidentale" in flora.public_domain_botanical_tags
    assert "INVERTED_GRAVITROPISM" in flora.mutation_tags

    # 4. Test VoxelboxterEngine public domain tag aggregation & world export
    engine.constructs["test_construct"] = graph
    world_state = engine.export_world_state()

    assert world_state["constructs_count"] == 1
    assert "public_domain_tags" in world_state
    assert "public_domain_geometry" in world_state["public_domain_tags"]
    assert "voxelboxter" in world_state["public_domain_tags"]
    assert "structural_honesty" in world_state["public_domain_tags"]
    assert world_state["structural_honesty"] is True


def test_orchestrator_arxiv_public_domain_tags():
    """Verify UniversalOrchestrator attaches public domain tags upon arXiv preprint ingestion."""
    from src.core.orchestrator import UniversalOrchestrator

    orchestrator = UniversalOrchestrator(dim=32)
    # Ingest arXiv preprint dyad without fossilizing to disk
    residues = orchestrator.ingest_arxiv_dyad(
        arxiv_id="2402.12345",
        title="Non-commutative Gyroidic Minimal Surfaces",
        abstract="We investigate minimal surfaces in non-Euclidean manifolds.",
        categories=["math.DG", "math.AP"],
        fossilize=False
    )

    assert residues is not None
    dyad = orchestrator.last_dyad
    assert dyad is not None
    assert "tags" in dyad.metadata
    assert "public_domain_tags" in dyad.metadata
    assert "math.dg" in dyad.metadata["public_domain_tags"]
    assert "public_domain" in dyad.metadata["public_domain_tags"]
    assert "arxiv" in dyad.metadata["tags"]
    assert dyad.metadata.get("structural_honesty") is True

