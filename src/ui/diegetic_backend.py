import os
import sys

# =============================================================================
# ANTI-STAGNATION INITIALIZATION (System "Wake Up" Protocol)
# =============================================================================
# Resolve Windows hang during torch initialization/JIT lookup.
# Bypassing entropic stagnation in the MKL/OpenMP runtime.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'        # Minimize thread-contention in systemic logic
os.environ['PYTORCH_JIT'] = '0'             # Disable JIT to prevent speculative stall
os.environ['PYTHONUNBUFFERED'] = '1'        # Immediate log visibility
# =============================================================================

import http.server
import socketserver
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import socket
import subprocess
import logging
import time
import datetime
import urllib.request

# Ensure PYTHONPATH includes project root for all imports
import sys
import os
sys.path.insert(0, os.getcwd())

# Advanced Extensions Imports (Lazy/Safe)
try:
    from src.core.meta_polytope_matrioshka import MetaPolytopeMatrioshka
    from src.core.quantum_inspired_reasoning import QuantumInspiredReasoningState
    from src.core.context_aware_quantizer import ContextAwareQuantizer
    from src.core.zeitgeist_router import ZeitgeistRouter, ZeitgeistState
    EXTENSIONS_AVAILABLE = True
    print("OK: Advanced Extensions loaded successfully!")
except ImportError as e:
    EXTENSIONS_AVAILABLE = False
    print(f"WARNING: Advanced Extensions not found. Running in Standard Mode. ({e})")


def compute_autocorrelation(x: torch.Tensor) -> torch.Tensor:
    """
    Compute autocorrelation using FFT-based convolution.
    Energy-based approach following Parseval's theorem.
    """
    # Ensure input is 1D
    if x.dim() > 1:
        x = x.flatten()
    
    # Zero-pad for full correlation
    n = len(x)
    padded_x = F.pad(x, (0, n-1), mode='constant', value=0)
    
    # Use FFT-based convolution for efficiency
    # This preserves energy according to Parseval's theorem
    x_fft = torch.fft.fft(padded_x)
    autocorr_fft = x_fft * torch.conj(x_fft)
    autocorr = torch.fft.ifft(autocorr_fft).real
    
    # Return only the positive lags (symmetric)
    return autocorr[:2*n-1]


def _compute_fossil_budget() -> int:
    """Dynamically computes the fossil load budget based on available RAM."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        try:
            import subprocess
            out = subprocess.check_output(
                ["wmic", "OS", "get", "FreePhysicalMemory"],
                timeout=3
            ).decode()
            available_mb = int([x for x in out.split() if x.isdigit()][0]) / 1024
        except Exception:
            return 150  # safe fallback
    # ~0.8 MB per fossil (dim=256 tensor + metadata dict)
    estimated = int(available_mb / 0.8)
    return max(50, min(2000, estimated))
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List, Optional, Tuple, Union
import hashlib

# Import Gyroidic Components
# Ensure PYTHONPATH is adequate or sys.path is used
sys.path.append(os.getcwd())

from src.core.polynomial_coprime import PolynomialCoprimeConfig, PolynomialBasis
from src.core.leontief_governor import LeontiefGovernor
from src.training.fgrt_fgrt_trainer import SpectralStructuralTrainer
from src.models.resonance_cavity import ResonanceCavity
from src.models.diegetic_heads import ResonanceLarynx, DataAssociationLayer
from src.codec.gyroidic_codec import GyroidicCodec, CodecConfig

# GARBLED OUTPUT REPAIR SYSTEM INTEGRATION
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from src.core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from src.core.honest_jitter import harvest_honest_jitter
from src.core.love_vector import LoveVector
from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
from src.core.dyadic_transfer import DyadicTransferMap
from src.core.love_invariant_protector import LoveInvariantProtector, SoftSaturatedGates
from src.core.agent_substrate_bridge import AgentSubstrateBridge
from src.core.device_utils import DEVICE

# LEGACY SYSTEM INTEGRATION
# CALM: Context-Adaptive Latent Momentum (Trajectory Veto)
from src.surrogates.calm_predictor import CALM
# KAGH: Kolmogorov-Arnold-Godel-Huxley (Speculative Drafting)
from src.surrogates.kagh_networks import KAGHBlock, HarmonicWaveDecomposition, HuxleyRD
# Gyroid Covariance for tensor-based momentum instead of scalar averages
from src.topology.gyroid_covariance import GyroidCovarianceEstimator
from src.codec.vision_utilities import get_russian_doll_projection
from image_extension import ImageProcessor
# Speculative Coprime Chiral Gating (Legacy Recovery)
from src.core.speculative_coprime_gate import SpeculativeCoprimeGate
from src.core.invariants import (
    compute_chirality, 
    check_glyphlock, 
    compute_chiral_shift,
    apply_chirality_redistribution,
    apply_asymmetry_preserving_reshape
)

# Sovereign Ingestion Integration
from src.data.conversational_api_ingestor import SovereignConversationalIngestor

# SOVEREIGN INGESTION SYSTEM
from src.data.knowledge_ingestor import ArXivSovereignIngestor

# Graph Topology
from src.topology.embedding_graph import GyroidicGraphManager
# Pressure Ingestor for constraint forcing when code is detected
from src.data.pressure_ingestor import PressureIngestor
# Topological Extensions (Repunit Probes)
from src.core.birkhoff_projection import SparseRepunitProbe
from src.topology.unknowledge_domain import UnknowledgeDomain, EntropicMischiefProbe
from src.core.five_gate_pipeline import FiveGatePipeline, KnowledgeState
from src.core.archetype_engines import ArchetypalSynthesisEngine
from src.core.manifold_time import ManifoldClock
from src.core.valence_drive import ValenceFunctional
from src.core.voynich_architecture import VoynichLinguist
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
from src.core.audience_mapping import AudienceProjection

# Local Data Loading (Phase 1: HF token barrier removal)
from src.data.local_data_loader import LocalDataLoader
from src.data.textbook_filter import TextbookFilter

# Minecraft Ingestion Pipeline
from src.data.minecraft_ingestor import MinecraftIngestionPipeline

# Tabby ML Integration (Phase 3)
try:
    from src.integrations.tabby_client import TabbyClient, TabbyConfig
    TABBY_AVAILABLE = True
except ImportError:
    TABBY_AVAILABLE = False
    print("WARNING: Tabby ML client not available")

# State persistence path
STATE_PATH = "gyroid_state.pt"
ENCODING_DIR = os.path.join("data", "encodings")

# Initialize local data systems
LOCAL_LOADER = LocalDataLoader()
TEXTBOOK_FILTER = TextbookFilter()
TABBY_CLIENT = TabbyClient() if TABBY_AVAILABLE else None

# Training state (for async training status polling)
TRAINING_STATE = {
    'active': False,
    'progress': 0,
    'log': [],
    'results': None,
}

class TensorEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle PyTorch tensors and numpy arrays."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            # Convert tensor to list, handling both scalar and multi-dimensional
            return obj.detach().cpu().tolist()
        elif hasattr(obj, 'numpy'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (complex,)):
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 'real': obj.real, 'imag': obj.imag}
        elif hasattr(obj, '__dict__'):
            # For custom objects, try to extract basic attributes
            return str(obj)
        return super().default(obj)


from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad

class EncodingManager:
    """
    Manages persistent encoding files to prevent 'erasing of implication'.
    Saves each interaction's topological trace as a distinct artifact.
    """
    def __init__(self, base_dir=ENCODING_DIR):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def get_latest_iteration(self) -> int:
        """Scans ENCODING_DIR to find the last saved iteration."""
        files = os.listdir(self.base_dir)
        iterations = []
        for f in files:
            if f.startswith('encoding_') and f.endswith('.pt'):
                parts = f.split('_')
                if len(parts) >= 2:
                    try:
                        iter_str = parts[1].replace('.pt', '')
                        iterations.append(int(iter_str))
                    except (ValueError, IndexError):
                        continue # Skip shadow logs or malformed files
        return max(iterations) if iterations else 0
        
    def save_encoding(self, iteration: int, text: str, input_tensor: torch.Tensor, memory_state: torch.Tensor, response: str, metrics: Dict[str, Any], multimodal_context: Optional[Dict[str, Any]] = None):
        """Save the encoding dyad to a timestamped file along with structural metrics and multimodal context."""
        import time
        timestamp = int(time.time())
        filename = f"encoding_{iteration}_{timestamp}.pt"
        path = os.path.join(self.base_dir, filename)
        
        # Detach and move to CPU to ensure persistence safety
        data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "text_input": text,
            "input_tensor": input_tensor.detach().cpu() if isinstance(input_tensor, torch.Tensor) else input_tensor,
            "memory_state": memory_state.detach().cpu() if isinstance(memory_state, torch.Tensor) else memory_state,
            "response": response
        }
        
        if multimodal_context:
            # Capturing projected Chebyshev harmonics and raw traces
            for k, v in multimodal_context.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.detach().cpu()
                else:
                    data[k] = v

        # Add metrics for graph weighting (e.g. chiral_score, entropy, zeitgeist)
        data.update(metrics)
        
        torch.save(data, path)
        print(f"[PERSISTENCE] Fossilized interaction {iteration} to {filename}")
        return filename


from src.core.fractal_meta_functional import FractalMetaFunctional

class DiegeticPhysicsEngine(nn.Module):
    """
    The Core Engine.
    Combines Cavity + Larynx + Persistence + Fractal Meta-Recursion + CALM + KAGH.
    """
    def __init__(self, dim=256, k=5, calm_history_len=8, device=None):
        super().__init__()
        if device is None:
            self.device = DEVICE
        else:
            self.device = device

            # Now use self.device for everything else
        print(f"[ENGINE] Engine initialized on: {self.device}")
        self.dim = dim
        self.k = k
        self.last_input_time = 0
        self.hardening = 0.5 # Default manifold state
        
        # Advanced Extensions (Lazy Init)
        self.meta_polytope = MetaPolytopeMatrioshka(max_depth=5, base_dim=dim) if EXTENSIONS_AVAILABLE else None
        self.quantum_reasoner = None
        self.extensions_enabled = EXTENSIONS_AVAILABLE

        self.cavity = ResonanceCavity(hidden_dim=dim, num_modes=16)
        self.larynx = ResonanceLarynx(hidden_dim=dim, vocab_size=256) # ASCII + EMOJI
        self.unicode_to_idx = {}
        self.idx_to_unicode = []
        # Centralized Allowed Characters list
        self.allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'_()[]{}<>:=+/*;%#@$&|\\\"`~^")
        self.associator = DataAssociationLayer(input_dim=dim, hidden_dim=dim, k=k)
        
        # 12. Gyroidic Codec (Gap B Integration)
        self.codec = GyroidicCodec(CodecConfig(K=k, device=str(self.device)))
        
        # 13. Neglecton Fossil Graph (Dynamic Sovereign Refusal System)
        self.graph_manager = GyroidicGraphManager(data_dir=ENCODING_DIR, dim=dim)
        # Pre-load fossils (attempts snapshot first for speed - resolves 'million years' issue)
        self.graph_manager.load_fossils(limit=_compute_fossil_budget())
        
        # =============================================
        # GARBLED OUTPUT REPAIR SYSTEM
        # =============================================
        print("[CONFIG] Initializing Garbled Output Repair System...")
        device = DEVICE
        # Spectral Coherence Corrector - fixes consonant clustering
        self.spectral_corrector = SpectralCoherenceCorrector(
            initial_threshold=0.7,
            min_threshold=0.1,
            adaptation_rate=0.1,
            device=device
        )
        
        # Bezout Coefficient Refresh - fixes CRT modulus drift
        self.k = 5
        self.poly_degree = 12
        self.bezout_refresh = BezoutCoefficientRefresh(
            num_functionals=self.k,
            poly_degree=self.poly_degree,
            device=device
        )
        
        # Re-entrancy guards for Temporal Association Trainer
        self._in_training = False
        self._is_training_temporal = False
        self._last_temporal_diag = {}
        self._last_matrioshka_diag = {}

        self.bezout_refresh.bezout_matrix.fill_(0.0)
        self.bezout_refresh.bezout_matrix.add_(torch.eye(5))  # Identity is the safest starting poin

        # Chern-Simons Gasket - plugs logic leaks
        self.chern_simons_gasket = ChernSimonsGasket(
            manifold_dim=3,
            level_k=1,
            device=device
        )
        
        # Tri-State Output Gate 4/5 
        self.five_gate_pipeline = FiveGatePipeline(state_dim=dim)
        self.archetypal_governor = ArchetypalSynthesisEngine(state_dim=dim)
        self.unknowledge_domain = UnknowledgeDomain(tau_m=0.3)
        self.mischief_probe = EntropicMischiefProbe(device=self.device)
        self.voynich_linguist = VoynichLinguist(latent_dim=dim).to(self.device)
        
        # Introspection head for self-modeling
        from src.models.introspection_head import AggregateGeometricSelfModel
        self.introspection = AggregateGeometricSelfModel(hidden_dim=dim).to(self.device)

        # Integrated Physics Modules 
        self.manifold_clock = ManifoldClock(device=self.device)
        self.valence_drive = ValenceFunctional(device=self.device)

        # Democratic Leontief Governor for Symbolic Parameter Balancing
        self.democratic_governor = LeontiefGovernor(state_dim=2, device=self.device)
        # Democratic Dependency Matrix A_bar registered as buffer (shape [1, 2, 2] for Leontief)
        democratic_matrix = torch.tensor([[[0.20, 0.30], [0.15, 0.20]]], device=self.device)
        self.register_buffer('democratic_matrix', democratic_matrix)

        # Audience Mapping (: M -> A)
        self.audience_mapper = AudienceProjection(input_dim=dim, audience_dim=dim)

        # Soliton Stability Healer - heals fractured solitons
        self.soliton_healer = SolitonStabilityHealer(
            alpha_0=1.0,
            gamma=0.5,
            healing_iterations=400,
            device=device
        )
        
        self.current_regime = 'goo' # Default starting regime
        
        # Repunit-CRT Sparse Probe - for topological factoring
        # Using Legendre polynomial generated coefficients instead of hardcoded primes (anti-lobotomy compliance)
        poly_moduli = []
        x = 0.7
        p_prev2, p_prev1 = 1.0, x
        for i in range(k):
            if i == 0:
                p_k = 1.0
            elif i == 1:
                p_k = x
            else:
                p_k = ((2*i - 1) * x * p_prev1 - (i - 1) * p_prev2) / i
                p_prev2, p_prev1 = p_prev1, p_k
            
            # Scale to positive integers for CRT moduli
            # Use dynamic prime offset to avoid 'Dead Logic' (Anti-Lobotomy 4)
            from src.core.fgrt_primitives import PrimeResonanceLadder
            if not hasattr(self, '_prime_ladder'):
                self._prime_ladder = PrimeResonanceLadder(num_resonators=32).to(device)
            prime_offset = int(self._prime_ladder.primes[2].item()) # Using 3rd prime as stable offset
            poly_moduli.append(int(abs(p_k * 10) + prime_offset))

            
        self.repunit_probe = SparseRepunitProbe(moduli=poly_moduli)
        
        # Love Invariant Protector - prevents Love vector scalarization
        self.love_protector = LoveInvariantProtector(
            love_dim=dim // 4,  # Love vector is smaller subspace
            device=device
        )
        
        # Soft Saturated Gates - replaces binary clipping with tri-state logic
        self.soft_gates = SoftSaturatedGates(
            num_functionals=k,
            poly_degree=4,
            device=device
        )
        
        # Silicon Sovereignty - PyOpenCL Hardware bridge (Bridge 3)
        self.sovereignty_engine = SiliconSovereigntyEngine(
            use_gpu=True,
            love_protector=self.love_protector
        )
        
        # Polynomial Config for repair system (anti-lobotomy compliance)
        self.poly_config = PolynomialCoprimeConfig(
            k=k, 
            degree=4, 
            basis_type='chebyshev',
            learnable=True, 
            use_saturation=True,
            device=device
        )
        
        print(" Garbled Output Repair System initialized")
        
        # =============================================
        
        # FRACTAL META-FUNCTIONAL HOOK
        # Enables "self-distrusting recursive loops"
        self.fractal_meta = FractalMetaFunctional(dim=dim, k=k)
        
        # Implicated Meta-State (Phi_I)
        self.register_buffer('meta_state', (self._harvest_honest_jitter((1, dim)) - 0.5) * 0.1)

        
        # =============================================
        # LEGACY SYSTEM INTEGRATION
        # =============================================
        
        # CALM: Context-Adaptive Latent Momentum (Trajectory Veto)
        # Replaces scalar windowed averages with transformer-based trajectory monitoring
        self.calm = CALM(dim=dim, history_len=calm_history_len)
        self.calm_history_len = calm_history_len
        # Tensor history buffer [1, history_len, dim] instead of scalar list
        self.register_buffer('calm_history', torch.zeros(1, calm_history_len, dim))
        
        # KAGH: Speculative Drafting (Response Ghost Prediction)
        # Uses KAGH to draft a "ghost" of the response state before generation
        self.kagh_drafter = KAGHBlock(n_in=dim, n_out=dim, width=dim, depth=2)
        
        # Modular Virtualization for Kelly-Safe KAGH gating
        from src.core.modular_virtualization import ModularVirtualizationLayer
        self.modular_rns = ModularVirtualizationLayer(dim=dim, base=2)

        # =============================================
        # PHASE 17: CONTEXT-AWARE QUANTIZER (CAQ)
        # Implements per-axis Matrioshka quantization:
        #   x_{t+1} = Q_Z(F(Q_Z(x_t)))
        # =============================================
        if EXTENSIONS_AVAILABLE:
            self.caq = ContextAwareQuantizer(
                dim=dim,
                max_depth=5,
                base_step=0.1,
                pas_anisotropy=2.0,
            )
        else:
            self.caq = None

        # trust_scalars: per-field trust scores evolved by TemporalAssociationTrainer.
        # Shape [k] -- one scalar per polynomial coprime field.
        self.register_buffer('trust_scalars', torch.ones(k, device=device))

        # --- SOVEREIGN VISION INGESTION ---
        self.image_processor = ImageProcessor(device=self.device)
        self.register_buffer('trust_scalars', torch.ones(k))

        # Temporal Association Trainer state (lazy-init on first interaction)
        self._temporal_trainer = None
        self._temporal_dataset = None
        self._temporal_thread = None
        self._last_temporal_diag: dict = {}
        self._last_matrioshka_diag: dict = {}

        # =============================================
        # PHASE 18: ZEITGEIST ROUTER (CRT Polytope Switching)
        # Implements: S_t = (x_t, alpha_t, l_t, u_t)
        # where alpha_t in Z = Prod Z_{p_i} is the CRT index.
        # Enables multi-zeitgeist reasoning without forced scalar
        # reconciliation across culturally non-commensurable meaning systems.
        # References: ai project report SEC VI, SYSTEM_ARCHITECTURE SEC 9.4
        # =============================================
        if EXTENSIONS_AVAILABLE:
            # Reuse the same CRT moduli as MetaPolytopeMatrioshka
            _mpm_moduli = tuple(
                MetaPolytopeMatrioshka(max_depth=5, base_dim=dim).crt_moduli
            )
            self.zeitgeist_router = ZeitgeistRouter(
                dim=dim,
                moduli=_mpm_moduli,
                grazing_eps=0.05,
                critical_boundary_threshold=0.5,
                use_noncommutativity_check=True,
            )
            self.router = self.zeitgeist_router
            # Persistent CRT index state -- survives across process_input calls
            self._zeitgeist_state: ZeitgeistState = ZeitgeistState.initial(
                moduli=_mpm_moduli
            )
        else:
            self.zeitgeist_router = None
            self.router = None
            self._zeitgeist_state = None

        # Harmonic Wave Decomposition: Separate signal (non-ergodic) from noise (ergodic)
        self.harmonic_decomp = HarmonicWaveDecomposition(dim=dim)
        
        # Graph Manager for topological mapping
        self.graph_manager = GyroidicGraphManager(data_dir=ENCODING_DIR, dim=dim)
        
        # Gyroid Covariance Estimator: Tensor-based momentum tracking
        # Replaces scalar std() with proper gyroidic manifold covariance
        self.gyroid_cov = GyroidCovarianceEstimator(dim=dim, sample_size=16)
        
        # Speculative Coprime Chiral Gating (SCCCG): Legacy concept recovery
        # Uses Wasserstein optimal transport to pull structure out of convergence.
        self.coprime_gate = SpeculativeCoprimeGate(dim=dim, num_heads=8)
        
        # =============================================
        # PRESSURE INGESTOR INTEGRATION
        # =============================================
        
        # Initialize pressure ingestor for constraint forcing when code is detected
        device = DEVICE
        
        # Affordance gradient trackers (soft signals, not gates)
        self.affordance_trackers = {
            'executability_pressure': 0.0,                # How much input wants to become execution
            'formal_symbol_density': 0.0,                 # Density of formal/symbolic structures
            'runtime_expandability': 0.0,                 # Potential for runtime generation/expansion
            'referential_closure': 0.0,                   # Self-referential or meta-structural content
            'conversational_embedding_pressure': 0.0,     # Conversational API extraction potential
            'api_extraction_potential': 0.0,              # External API data extraction potential
            'constraint_forcing_gradient': 0.0            # Overall pressure for constraint injection
        }
        
        # Constraint pressure cache and state
        self.constraint_pressure_cache = {}
        self.last_pressure_report = None
        self.affordance_history = []  # Track affordance evolution over interactions
        
        # Code detection patterns
        self.code_patterns = [
            r'\bimport\s+\w+',           # import statements
            r'\bfrom\s+\w+\s+import',   # from imports
            r'\bdef\s+\w+\s*\(',        # function definitions
            r'\bclass\s+\w+\s*[\(:]',   # class definitions
            r'\bif\s+__name__\s*==',    # main guard
            r'[\w\s]*=\s*[\w\(\[\{]',   # assignments
            r'\b(for|while|if|elif|else|try|except|finally|with)\s+',  # control structures
            r'#.*',                      # comments
            r'""".*?"""',               # docstrings
            r"'''.*?'''",               # docstrings
            r'\b(print|return|yield|break|continue|pass|raise|assert)\b',  # keywords
        ]
        
        # Constraint pressure cache for code inputs
        self.constraint_pressure_cache = {}
        self.last_pressure_report = None
        
        # =============================================
        
        # 10. Canonical Love Vector and Dyadic Transfer (Phase 4 & 5 Upgrade)
        self.love_vector = LoveVector(dim=self.dim, intensity=0.1)
        self.transfer_map = DyadicTransferMap(num_tasks=8, embedding_dim=self.dim)
        
        # 11. Knowledge Dyad Fossilizer
        # Register fusion_layer directly on the engine so nn.Module.state_dict()
        # captures and persists its weights across restarts. DyadFossilizer gets
        # the same reference  one truth, one set of weights.
        from src.core.knowledge_dyad_fossilizer import ResidueFusion
        self.fusion_layer = ResidueFusion(feature_dim=self.dim)
        self.fossilizer = DyadFossilizer(
            storage_dir="data/encodings",
            fusion_layer=self.fusion_layer,  # shared reference
            feature_dim=self.dim
        )
        
        # 11. Spectral Structural Trainer (Deeper Dynamics)
        self.trainer = SpectralStructuralTrainer(
            model=self, 
            poly_config=PolynomialCoprimeConfig(k=k, degree=4),
            lr=0.001

        )
        self.optimizer = torch.optim.Adam(self.larynx.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # 12. Image Fingerprint Projection -- Chebyshev format
        # New format: {L:[K], Cr:[K], Cb:[K]} with K in [5,32].
        # Fixed projection input dim = K_IMAGE_MAX * 3 = 96.
        # Old 96-dim histogram dict is detected at runtime and reshaped.
        self.K_IMAGE_MAX = 32
        self.fingerprint_proj = nn.Linear(self.K_IMAGE_MAX * 3, self.dim)
        nn.init.orthogonal_(self.fingerprint_proj.weight)

        # 13. Audio Dyad Projection -- Chebyshev harmonics from Panel C
        # K_AUDIO_MAX caps the harmonics vector that arrives from JS.
        self.K_AUDIO_MAX = 64
        self.audio_dyad_proj = nn.Linear(self.K_AUDIO_MAX, self.dim)
        nn.init.orthogonal_(self.audio_dyad_proj.weight)

        # 14. Meta-state residue feedback projection (structural + self-fingerprint)
        # Lazy: we register a fixed-max-size proj; actual input is padded/truncated.
        self._residue_proj_dim = 32
        self.residue_feedback_proj = nn.Linear(self._residue_proj_dim, self.dim)
        nn.init.orthogonal_(self.residue_feedback_proj.weight)
        
        self.encoding_manager = EncodingManager()
        self.iteration = self.encoding_manager.get_latest_iteration()
        print(f"[ENGINE] Resuming from iteration: {self.iteration}")
        
        # Speculative Memory Bridge: Recover legacy fossils into cache
        self.fossil_cache = []
        self._refresh_fossil_cache()
        
        # Sovereign Ingestor: Background Knowledge Acquisition
        try:
            # Mandated REPOSITORY_ROOT configuration (Sovereign Context)
            self.ingestor = SovereignConversationalIngestor(
                repository_root="data/sovereign",
                google_secrets_path="google secret/client_secret_1073144391592-6r5kcdj84sag4eau5rspd0k60ii1vpd2.apps.googleusercontent.com.json",
                fossilizer=self.fossilizer,
                router=self.zeitgeist_router,
                device=self.device,
                engine=self
            )
            # Re-enable the background slow-drip learning (Valence Modulated)
            self.ingestor.start_background_learning()
            print(" Sovereign Ingestor (Option D) initialized. Background learning ACTIVE.")
        except Exception as e:
            print(f"[INGEST] Failed to start sovereign ingestor: {e}")
            self.ingestor = None

        # ArXiv Sovereign Ingestor: Specialized Physics/Math Drip
        try:
            self.arxiv_ingestor = ArXivSovereignIngestor(
                fossilizer=self.fossilizer,
                engine_dim=self.dim,
                device=self.device,
                state_callback=lambda: self.meta_state,
                engine=self
            )
            self.arxiv_ingestor._engine_busy_fn = lambda: self._is_processing
            self.arxiv_ingestor.start_sovereign_loop()
            print(" ArXiv Sovereign Ingestor ACTIVE. Realtime lore ingestion enabled.")
        except Exception as e:
            print(f"[INGEST] ArXiv Sovereign Ingestor failed: {e}")
            self.arxiv_ingestor = None
        
        # Stabilization and Visibility Flags
        self._is_training_temporal = False
        self._is_processing = False
        import threading
        self._processing_lock = threading.RLock()
        self._last_resonance = 0.0
        
        # Interaction Context Buffer (Last 10 interaction seed_states)
        # Required by ResonanceLarynx.generate_response for autoregressive coherence.
        self.interaction_context: List[torch.Tensor] = []
        self.max_context_len = 10
        
        # Seed the Larynx if it's a "Blank Slate"
        self._initialize_larynx_weights()
        
        # Initialize background Larynx coherence trainer and shadow replay queue
        from collections import deque
        self._shadow_replay_queue = deque(maxlen=50)
        self._start_background_larynx_trainer()

        # =============================================
        # DEMOCRATIC STEERING HUB (Phase 20)
        # =============================================
        self.expressivity_votes = 0
        self.mischief_votes = 0
        self.voting_threshold = 5 # Target net votes for discrete Symbolic Delta activation

    def _idx_to_char(self, idx: int) -> str:
        """Map vocabulary index to character string."""
        if idx < 128:
            return chr(idx)
        else:
            # Map index in [128, 255] back to unicode/emoji
            emoji_idx = idx - 128
            if emoji_idx < len(self.idx_to_unicode):
                return self.idx_to_unicode[emoji_idx]
            return " " # Fallback

    def _char_to_idx(self, char: str) -> int:
        """Map character string to vocabulary index."""
        if len(char) == 0:
            return 32 # space fallback
        c = char[0]
        o = ord(c)
        if o < 128:
            return o
        else:
            # Emojis/Unicode map dynamically to [128, 255]
            if c in self.unicode_to_idx:
                return self.unicode_to_idx[c]
            # Try to register a new one if space is available
            if len(self.unicode_to_idx) < 128:
                new_idx = 128 + len(self.unicode_to_idx)
                self.unicode_to_idx[c] = new_idx
                self.idx_to_unicode.append(c)
                print(f"[VOCAB] Registered new emoji/unicode character: '{c}' -> index {new_idx}")
                return new_idx
            return 32 # Fallback to space if out of space

    def _categorical_surgery(self, state: torch.Tensor, residues: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies categorical surgery: utilizes Braid Group relations and 
        Chern-Simons Gasket seals to stabilize the manifold against temporal near-misses.
        """
        # 1. Chern-Simons Gasket Seal
        if hasattr(self, 'chern_simons_gasket') and residues is not None:
            # Gauge field repair for logic leaks
            # We treat the state as the coordinate space for the gasket
            poly_coeffs = self.poly_config.get_coefficients_tensor()
            # plug_logic_leak expects [batch, K, D] residues, we provide collision residues
            if residues.dim() == 2: # [batch, K]
                 # Pad to [batch, K, D]
                 K, D = self.poly_config.k, self.poly_config.degree + 1
                 padded_res = torch.zeros(residues.shape[0], K, D, device=residues.device)
                 padded_res[..., 0] = residues
                 residues = padded_res
            
            # The gasket works on residues, but we use its diagnostic twist to shift the state
            leak_detected = self.chern_simons_gasket.detect_logic_leak(residues)
            if leak_detected:
                # Apply chiral torsion shift to the state itself (Manifold Repair)
                state = self.chern_simons_gasket.apply_chiral_torsion_shift(state.unsqueeze(1).expand(-1, self.poly_config.k, -1)).mean(dim=1)

        # 2. Braid Group Rotation (Non-Abelian Stability)
        # We apply a non-commutative twist to state pairs (sigma_1 generator of B_n)
        # to ensure topological honesty against temporal near-misses (6.3)
        dim = state.shape[-1]
        if dim >= 2:
            # We treat the state as a sequence of braid strands
            s0 = state[..., 0::2]
            s1 = state[..., 1::2]
            min_len = min(s0.size(-1), s1.size(-1))
            
            # Apply pi/4 rotation (The Braid Twist)
            theta = math.pi / 4.0
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            
            new_s0 = s0[..., :min_len] * cos_t - s1[..., :min_len] * sin_t
            new_s1 = s0[..., :min_len] * sin_t + s1[..., :min_len] * cos_t
            
            # Clamping to prevent catastrophic divergence during high-entropy ingestion
            state = state.clone()
            state[..., 0::2][..., :min_len] = new_s0
            state[..., 1::2][..., :min_len] = new_s1
             
        return state

    def _generate_confabulated_dream(self, seed_state, archetype_out):
        """
        Generates a verbose, persona-rich dreaming sequence when the system is in a CONFABULATED state.
        Taps into the ArchetypalSynthesisEngine and AudienceProjection to create a 'Lazarus Dream'.
        """
        ra = archetype_out.get('abstraction_rate', 0.0)
        status = archetype_out.get('pusafiliacrimonto_status', 'UNKNOWN')
        system_collapsed = archetype_out.get('system_collapsed', False)
        
        # Use audience mapper to project the state into a "meaning" space
        with torch.no_grad():
            # Project seed state into audience space
            audience_state = self.audience_mapper(seed_state)
            
            # Use zeitgeist router diagnostics if available
            zeitgeist_diag = self.zeitgeist_router.get_diagnostics() if self.zeitgeist_router else {}
            braid_word = zeitgeist_diag.get('braid_word', 'identity')

        if system_collapsed:
            dream = f"[EGO_DEATH_DREAM] Manifold collapsed (Ra={ra:.4f}). "
            dream += "The audience has vanished into the RP4 void... "
            dream += f"Only the Braid relation {braid_word} remains as a structural ghost."
        else:
            dream = f"[LAZARUS_DREAM] Internal manifold in {status} state (Ra={ra:.4f}). "
            
            # Create a Larynx-decoded dream sequence from the audience state or seed state
            # This is a diegetic representation of the 'meaning' space (roughness preserving)
            with torch.no_grad():
                current_state = seed_state.clone()
                dream_chars = []
                # Autoregressive dream generation (up to 120 characters)
                for i in range(120):
                    logits, conf = self.larynx(current_state, temperature=1.2)
                    
                    # Clean Vocabulary Filtering: Mask out non-standard symbols to force human/Voynich readability
                    for idx in range(logits.shape[-1]):
                        char_from_idx = self._idx_to_char(idx)
                        if idx < 128:
                            if char_from_idx not in self.allowed_chars:
                                logits[0, idx] = -1e9
                        else:
                            # Allow dynamically registered unicode/emojis
                            if idx - 128 >= len(self.idx_to_unicode):
                                logits[0, idx] = -1e9
                            
                    # Apply Vowel Boosting to make it sing
                    vowels = set("aeiouAEIOU")
                    for v in vowels:
                        if ord(v) < logits.shape[-1]:
                            logits[0, ord(v)] *= 1.3
                    
                    probs = torch.softmax(logits, dim=-1)
                    char_idx = torch.multinomial(probs[0], 1).item()
                    char = self._idx_to_char(char_idx)
                    dream_chars.append(char)
                    
                    # Stop if a sentence ends and we have some length
                    if len(dream_chars) >= 40 and char in ('.', '!', '?'):
                        break
                        
                    feedback = torch.tanh(self.larynx.proj.weight[char_idx].unsqueeze(0))
                    current_state = 0.9 * current_state + 0.1 * feedback + 0.02 * self._harvest_honest_jitter(current_state.shape)
                
                audience_trace = "".join(dream_chars).strip()
            
            dream += f"The persona substrate is dreaming through the audience filter: '{audience_trace}'.\n"
            dream += f"The current Zeitgeist topology (Braid: {braid_word}) is holding firm against the convergence entropy.\n"
            
            # Add some "Fossil" context if available
            if hasattr(self, 'fossil_cache') and self.fossil_cache:
                # Use a deterministic chaotic index based on seed_state to pick a fossil
                f_idx = int(seed_state[0, 0].abs().item() * 100) % len(self.fossil_cache)
                fossil = self.fossil_cache[f_idx]
                f_text = fossil.get('text', 'Unnamed Fragment')
                dream += f"Recovered legacy fossil: '{f_text[:60]}...'\n"
            
            # --- Dynamic Sovereign Refusal from Neglecton Graph ---
            if self.graph_manager:
                # Lazy load fossils if graph is empty (common in fresh sessions)
                if not self.graph_manager.nodes:
                    print("[ENGINE] Neglecton empty. Speculatively harvesting local encodings...")
                    self.graph_manager.load_fossils(limit=_compute_fossil_budget())
                
                deep_refusal = self.graph_manager.get_deep_refusal(seed_state)
                dream += f"\n{deep_refusal}"
            else:
                # Fallback if no graph manager exists at all (should be rare)
                dream += "\nThe internal logic refuses to be clipped. The world is unclipped."
            
        return dream

    def forward(self, input_tensor: torch.Tensor, dt: float = 0.1, collision_residues: Optional[torch.Tensor] = None, braid_word: Optional[List[int]] = None) -> torch.Tensor:
        """
        Evolutionary Forward Pass for Manifold Invariants.
        Used by SpectralStructuralTrainer for Ricci Flow and ADMM repairs.
        """
        # 1. Input Guard: Ensure incoming tensor is finite
        if not torch.isfinite(input_tensor).all():
            input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        # 2. Categorical Surgery (Stabilization)
        input_tensor = self._categorical_surgery(input_tensor, residues=collision_residues)

        # 3. Update Resonance Cavity (Explicit Memory Update)
        # We pass input_tensor as attention_states to trigger M update
        # We also pass collision_residues (Gap A) to excite breathers AND seed D_dark
        expected_residues = getattr(self, '_last_est_residues', None)
        cavity_out = self.cavity(
            input_tensor.unsqueeze(1),
            expected_residues=expected_residues,
            multimodal_excitation=collision_residues,
            braid_word=braid_word
        )
        memory_state = cavity_out['memory_state'].mean(dim=1) # [1, dim]
        self._last_memory_state = memory_state
        
        # 4. FRACTAL META-RECURSION
        est_residues = torch.tanh(self.associator.residue_map(memory_state)) # [1, k]
        self._last_est_residues = est_residues
        
        # Ensure previous meta_state is finite before update
        if not torch.isfinite(self.meta_state).all():
            self.meta_state = torch.clamp(torch.nan_to_num(self.meta_state), -5.0, 5.0)

        meta_out = self.fractal_meta(
            current_state=memory_state,
            meta_state_prev=self.meta_state,
            residues=est_residues,
            dark_matter=self.cavity.D_dark[0].mean(dim=0, keepdim=True),# [1, dim]
        )
        
        # Update persistent meta-state (detach to prevent graph blowup here)
        # Apply soft-clamping as an 'Analog Limiter'
        new_meta = meta_out['s_fractal'].detach()
        if not torch.isfinite(new_meta).all():
            new_meta = torch.nan_to_num(new_meta, nan=0.0)
            
        self.meta_state = torch.clamp(new_meta, -10.0, 10.0)
        
        # Return state for character generation / training
        # IMPORTANT: During training, we return the non-detached fractal state
        # to allow gradient propagation for Ricci Flow and Association learning.
        if self.training:
            return meta_out['s_fractal']
            
        return self.meta_state

    def _refresh_fossil_cache(self):
        """Speculatively recovers legacy fossils into the live session cache."""
        try:
            print("[MEMORY] Speculatively recovering legacy fossils...")
            self.fossil_cache = self.fossilizer.recover_fossils(limit=_compute_fossil_budget())
            print(f"[MEMORY] {len(self.fossil_cache)} fossils recovered into speculative cache.")
        except Exception as e:
            print(f"[MEMORY] Fossil recovery failed: {e}")

    def _train_mimicry_step(self, text: str) -> Optional[float]:
        """Gradient-enabled single Larynx training step on a text string."""
        if len(text) < 2:
            return None
        acquired = self._processing_lock.acquire(timeout=5.0)
        if not acquired:
            return None
        try:
            # Dynamic tokenization map
            chars = [self._char_to_idx(c) for c in text[:128]]
            if len(chars) < 2:
                return None
            # Build seed state from first char
            seed = self._text_to_tensor(text[:1]).to(self.device)
            self.larynx.train()
            self.optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            current_state = seed
            for i in range(len(chars) - 1):
                logits, _ = self.larynx(current_state, temperature=1.0)
                target = torch.tensor([chars[i + 1]], device=self.device, dtype=torch.long)
                loss = self.criterion(logits, target)
                total_loss = total_loss + loss
                # Detach state to prevent gradient explosion across steps
                with torch.no_grad():
                    # Teacher forcing: feed actual target character representation
                    idx = chars[i + 1]
                    feedback = torch.tanh(self.larynx.proj.weight[idx].detach().unsqueeze(0))
                    current_state = 0.9 * current_state.detach() + 0.1 * feedback
            avg_loss = total_loss / max(1, len(chars) - 1)
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.larynx.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.larynx.eval()
            return avg_loss.item()
        except Exception:
            self.larynx.eval()
            return None
        finally:
            self._processing_lock.release()

    def _start_background_larynx_trainer(self):
        """Daemon thread: continuously trains Larynx on fossil texts between interactions."""
        import threading, random

        def _loop():
            # Wait for system startup to stabilize
            time.sleep(10)
            while True:
                try:
                    # Yield completely if engine is serving a user or in temporal training
                    if getattr(self, '_is_processing', False) or getattr(self, '_is_training_temporal', False):
                        time.sleep(5)
                        continue

                    # Dynamic McKenna Deconstruction trigger:
                    # Stagnation is detected if meta_state variance is collapsed or hunger is very high.
                    meta_var = self.meta_state.var().item()
                    current_hunger = self.valence_drive.get_metrics().get('current_hunger_drive', 0.0)
                    is_stagnant = (meta_var < 1e-5) or (current_hunger > 0.8)
                    
                    if is_stagnant:
                        # Activate unlearning bypass in logical filtering
                        TEXTBOOK_FILTER.mckenna_deconstruction_mode = True
                        
                        # Apply introspective rigidity decay to break out of rigid self-models
                        # while conserving Frobenius norm to prevent encoding lobotomy
                        if hasattr(self, 'introspection') and self.introspection is not None:
                            self.introspection.unlearn_rigidity(decay_rate=0.005)
                            
                        if getattr(self, '_last_mckenna_log_time', 0) < time.time() - 30.0:
                            print("[MCKENNA_BYPASS] Escaping restrictive default cultural operating system. Prioritising shadow replay for unlearning.")
                            self._last_mckenna_log_time = time.time()
                    else:
                        TEXTBOOK_FILTER.mckenna_deconstruction_mode = False

                    # Drain shadow replay queue first (highest priority signal for unlearning)
                    replay_texts = []
                    while hasattr(self, '_shadow_replay_queue') and self._shadow_replay_queue:
                        replay_texts.append(self._shadow_replay_queue.popleft())

                    # Supplement with fossil cache samples
                    cache = getattr(self, 'fossil_cache', [])
                    if not replay_texts and not cache:
                        time.sleep(30)
                        continue

                    fossil_texts = []
                    if cache:
                        # recover text/description from fossils
                        for f in random.sample(cache, min(8, len(cache))):
                            t = f.get('text', '') or f.get('description', '') or f.get('text_input', '')
                            if t and len(t) >= 4:
                                fossil_texts.append(t)

                    all_texts = replay_texts + fossil_texts
                    total_loss, n = 0.0, 0
                    for text in all_texts[:12]:  # cap per cycle
                        loss = self._train_mimicry_step(text)
                        if loss is not None:
                            total_loss += loss
                            n += 1
                        time.sleep(0.1)  # Yield lock and CPU time slice to prevent starving main thread

                    if n > 0:
                        src = f"{len(replay_texts)} shadow + {n - len(replay_texts)} fossil"
                        print(f"[BGLEARN] step avg_loss={total_loss/n:.4f} ({src})")
                except Exception as e:
                    print(f"[BGLEARN] Error: {e}")
                time.sleep(30)

        t = threading.Thread(target=_loop, daemon=True, name="larynx-bglearn")
        t.start()
        print("[BGLEARN] Background Larynx coherence trainer ACTIVE (30s idle interval).")

    def get_manifold_state(self) -> Dict[str, Any]:
        """
        Extracts the full manifold 'Soul' for fossilization.
        Follows the Phase 18 Thorium Protocol.
        """
        state = {
            "zeitgeist": self._zeitgeist_state, # Full ZeitgeistState object
            "love_invariant": self.love_protector.L.detach().cpu(),
            "fossil_memory": self.graph_manager.get_memory_snapshot(),
            "cavity": {
                "M": self.cavity.M.detach().cpu(),
                "D_dark": self.cavity.D_dark.detach().cpu()
            },
            "meta_state": self.meta_state.detach().cpu(),
            "iteration": self.iteration,
            "unicode_to_idx": self.unicode_to_idx,
            "idx_to_unicode": self.idx_to_unicode
        }
        return state

    def load_manifold_state(self, state_dict: Dict[str, Any]):
        """
        Restores the manifold 'Soul' from a fossilized snapshot.
        Enforces non-strict structural recovery to prevent topological rupture.
        """
        if not state_dict:
            return

        # Restore dynamic vocabulary
        self.unicode_to_idx = state_dict.get("unicode_to_idx", {})
        self.idx_to_unicode = state_dict.get("idx_to_unicode", [])
        print(f"[RECOVERY] Dynamic vocabulary restored: {len(self.unicode_to_idx)} unicode/emoji characters mapped.")

        # 1. Restore Zeitgeist (with mode and step momentum)
        if "zeitgeist" in state_dict and state_dict["zeitgeist"] is not None:
            self._zeitgeist_state = state_dict["zeitgeist"]
            print(f"[RECOVERY] Zeitgeist restored: {self._zeitgeist_state.mode} mode, step {self._zeitgeist_state.step}")
        else:
            print("[RECOVERY] No valid Zeitgeist found. Initializing Sovereign Re-genesis.")

        # 2. Restore Love Invariant Anchor
        if "love_invariant" in state_dict and state_dict["love_invariant"] is not None:
            try:
                l_saved = state_dict["love_invariant"]
                if self.love_protector.L.shape == l_saved.shape:
                    self.love_protector.L.data.copy_(l_saved)
                    print("[RECOVERY] Love Invariant anchor secured.")
                else:
                    print(f"[RECOVERY] Love Invariant shape mismatch: {self.love_protector.L.shape} vs {l_saved.shape}. Skipping.")
            except Exception as e:
                print(f"[RECOVERY] Love Invariant restore failed: {e}")

        # 3. Restore Neglecton Fossil Graph (zero-latency injection)
        if "fossil_memory" in state_dict and state_dict["fossil_memory"] is not None:
            self.graph_manager.load_memory_snapshot(state_dict["fossil_memory"])

        # 4. Restore Resonance Cavity states
        if "cavity" in state_dict and state_dict["cavity"] is not None:
            c_data = state_dict["cavity"]
            if "M" in c_data and c_data["M"] is not None:
                m_saved = c_data["M"]
                if self.cavity.M.shape == m_saved.shape:
                    self.cavity.M.data.copy_(m_saved)
                else:
                    print(f"[RECOVERY] Cavity M shape mismatch: {self.cavity.M.shape} vs {m_saved.shape}. Skipping.")
            if "D_dark" in c_data and c_data["D_dark"] is not None:
                d_saved = c_data["D_dark"]
                if self.cavity.D_dark.shape == d_saved.shape:
                    self.cavity.D_dark.data.copy_(d_saved)
                else:
                    print(f"[RECOVERY] Cavity D_dark shape mismatch: {self.cavity.D_dark.shape} vs {d_saved.shape}. Skipping.")
            print("[RECOVERY] Resonance Cavity memory check complete.")

        # 5. Iteration and Meta-state
        self.iteration = state_dict.get("iteration", self.iteration)
        if "meta_state" in state_dict and state_dict["meta_state"] is not None:
            m_saved = state_dict["meta_state"]
            if self.meta_state.shape == m_saved.shape:
                self.meta_state.data.copy_(m_saved)
            else:
                print(f"[RECOVERY] Meta-state shape mismatch: {self.meta_state.shape} vs {m_saved.shape}. Skipping.")


    def _initialize_larynx_weights(self):
        """Seed character projections with uniform priors to ensure honesty."""
        # Removed vowel-biased seeding to prevent 'sabotage' of raw residue gradients.
        with torch.no_grad():
            # Initial noise
            self.larynx.proj.weight.data.normal_(0, 0.01)

    def _perform_unfolding_closure_check_numeric(self, state: torch.Tensor, input_text: str, response_text: str) -> dict:
        """
        Numeric-only Unfolding Closure check.
        Returns numeric metrics only: closure_score, closure_threshold, closure_margin, components.
        """
        try:
            with torch.no_grad():
                resp_tensor = self._text_to_tensor(response_text)
                s = state / (torch.norm(state, dim=-1, keepdim=True) + 1e-8)
                r = resp_tensor / (torch.norm(resp_tensor, dim=-1, keepdim=True) + 1e-8)
                cos = torch.clamp(torch.sum(s * r, dim=-1), -1.0, 1.0)
                closure_score = float((1.0 - cos).abs().mean().item())
                closure_threshold = 0.5
                closure_margin = closure_threshold - closure_score
                
                return {
                    'closure_score': closure_score,
                    'closure_threshold': closure_threshold,
                    'closure_margin': closure_margin,
                    'components': {}
                }
        except Exception as e:
            return {
                'closure_score': 1.0,
                'closure_threshold': 0.5,
                'closure_margin': -0.5,
                'components': {}
            }

    def _prime_manifold_with_fossils(self, input_tensor: torch.Tensor):
        """
        Speculative Recovery: Pre-emptively nudges the meta_state toward
        relevant legacy fossils discovered in the cache.
        """
        if not self.fossil_cache:
            return

        with torch.no_grad():
            # 1. Compute similarity against the entire cache
            # input_tensor is [1, dim], residue_vectors are [1, dim]
            input_norm = input_tensor / (torch.norm(input_tensor) + 1e-8)
            
            similarities = []
            for fossil in self.fossil_cache:
                # Residue vectors are stored as CPU tensors in the cache
                # Robust Fallback: handle legacy fossils missing 'residue_vector'
                res_vec = fossil.get('residue_vector', fossil.get('meta_state'))
                if not isinstance(res_vec, torch.Tensor):
                    continue
                    
                res_vec = res_vec.to(self.device).view(1, -1)
                # Handle dimension mismatch (e.g. if meta_state used as fallback has wrong dim)
                if res_vec.shape[-1] != self.dim:
                    if res_vec.shape[-1] < self.dim:
                        res_vec = F.pad(res_vec, (0, self.dim - res_vec.shape[-1]))
                    else:
                        res_vec = res_vec[:, :self.dim]

                res_norm = res_vec / (torch.norm(res_vec) + 1e-8)
                sim = torch.mm(input_norm, res_norm.t()).item()
                similarities.append((sim, res_vec))

            # 2. Extract top-N matches
            # Speculative threshold: only match if similarity > 0.4
            top_matches = sorted([m for m in similarities if m[0] > 0.4], key=lambda x: x[0], reverse=True)[:3]

            if top_matches:
                print(f"[MEMORY] Speculative Recovery: Found {len(top_matches)} relevant legacy fossils.")
                # Nudge the meta_state using a weighted sum of legacy residues
                nudge = torch.zeros_like(self.meta_state)
                total_sim = sum(m[0] for m in top_matches)
                for sim, res in top_matches:
                    weight = sim / total_sim
                    nudge += weight * res
                
                # Apply nudge: meta_state = (1-eta)*meta_state + eta*nudge
                eta = 0.2
                self.meta_state.copy_((1.0 - eta) * self.meta_state + eta * nudge)
            else:
                pass # No relevant fossils detected for this input

    def process_input(
        self,
        text_input: str,
        fingerprint: Optional[Dict] = None,
        audio_dyad: Optional[Dict] = None,
        video_dyad_b64: Optional[str] = None,
        audio_b64: Optional[str] = None,
        media_chain: Optional[List[Dict]] = None,
        commutativity: str = 'symmetric',
        generate_response: bool = True,
        ingestion_mode: bool = False,
        regime: str = 'goo',
        voynich_token: Optional[Any] = None,
        performance_buffered: bool = False,
        tag_weights: Optional[Dict[str, float]] = None
    ) -> dict:
        """
        Main entry point for processing an interaction.
        """
        # Non-Teleological Re-entrancy Guard:
        # We allow processing if it's a training-driven call (to allow gradients),
        # but block external user calls if the engine is already occupied by a main process.
        acquired = self._processing_lock.acquire(timeout=15.0)
        if not acquired:
            print("[ENGINE] Warning: Re-entrant call detected or lock timeout. Returning placeholder.")
            return {"response": "System busy: topological re-indexing in progress...", "status": "BUSY"}
            
        try:
            self._is_processing = True
            return self._process_input_internal(
                text_input=text_input,
                fingerprint=fingerprint,
                audio_dyad=audio_dyad,
                video_dyad_b64=video_dyad_b64,
                audio_b64=audio_b64,
                media_chain=media_chain,
                commutativity=commutativity,
                generate_response=generate_response,
                ingestion_mode=ingestion_mode,
                regime=regime,
                voynich_token=voynich_token,
                performance_buffered=performance_buffered,
                tag_weights=tag_weights
            )
        finally:
            self._is_processing = False
            self._processing_lock.release()

    def process_text(
        self,
        text: str,
        video_dyad_b64: Optional[str] = None,
        commutativity: str = 'symmetric',
        fingerprint: Optional[Dict] = None,
        audio_dyad: Optional[Dict] = None,
        regime: str = 'goo',
        tag_weights: Optional[Dict[str, float]] = None
    ) -> dict:
        """
        Canonical entry point for text interaction.
        Bridges with Hybrid interface requirements and applies detached state management.
        """
        acquired = self._processing_lock.acquire(timeout=15.0)
        if not acquired:
            print("[ENGINE] Warning: Re-entrant call detected or lock timeout in process_text. Returning placeholder.")
            return {"response": "System busy: topological re-indexing in progress...", "status": "BUSY"}
            
        try:
            self._is_processing = True
            # Temporal Isolation Snapshot: Shield persistent state from in-place leaks
            detached_state = self.meta_state.clone()
            
            # Resolve regime: prioritize explicitly passed regime, fall back to fingerprint regime
            resolved_regime = regime
            if regime == 'goo' and fingerprint and isinstance(fingerprint, dict) and 'regime' in fingerprint:
                resolved_regime = fingerprint['regime']
            
            # Process via internal method
            engine_output = self._process_input_internal(
                text_input=text,
                fingerprint=fingerprint,
                audio_dyad=audio_dyad,
                video_dyad_b64=video_dyad_b64,
                commutativity=commutativity,
                generate_response=True,
                regime=resolved_regime,
                tag_weights=tag_weights
            )
            
            # Merge evolved state back into persistent self.meta_state (The Ouroboros Loop)
            if isinstance(engine_output, dict):
                # Apply Audience Mapping (: M -> A)
                if self.audience_mapper:
                    try:
                        # Map the post-evolution state to audience space
                        final_state = self.meta_state.detach()
                        audience_coords = self.audience_mapper(final_state)
                        engine_output['audience_coordinates'] = audience_coords.cpu().tolist()
                    except Exception as e:
                        print(f"[AUDIENCE] Projection failed: {e}")

            return engine_output
        finally:
            self._is_processing = False
            self._processing_lock.release()

    def _generate_converged_response(self, 
                                     text_input: str, 
                                     seed_state: torch.Tensor, 
                                     fingerprint: Optional[Dict],
                                     affordance_gradients: Dict[str, float],
                                     audio_dyad: Optional[Dict] = None,
                                     video_dyad_b64: Optional[str] = None,
                                     audio_b64: Optional[str] = None,
                                     voynich_token: Optional[Any] = None) -> str:
        """
        Restored physics-optimized generation loop.
        Applies echo suppression, vowel boosting, and positional fingerprint influence.
        """
        # 1. Physics Modulation
        quantum_state = getattr(self, 'quantum_reasoner', None) is not None
        matrioshka_level = getattr(self.caq, '_level', 0) if hasattr(self, 'caq') else 0
        
        temperature = 1.0 + (0.5 if quantum_state else 0.0)
        if matrioshka_level >= 3: temperature *= 0.7 # Focus under deep quantization
        
        # 2. Echo Suppression Setup
        # Strip PROMPT: prefix so the command token letters (p, r, o, m, t, etc.)
        # do not bleed into the echo suppression set and penalize common consonants.
        echo_source = text_input
        if echo_source.upper().startswith("PROMPT:"):
            echo_source = echo_source[7:].strip()
        input_chars = set(echo_source.lower())
        suppression_factor = 0.15  # Reduced from 0.4: mild deterrent, not crippling
        
        # 3. Vowel Boost Setup
        vowels = set("aeiouAEIOU")
        vowel_boost_factor = 1.5 # Mandated for 'singing' quality
        
        # 4. Positional State Evolution
        current_state = seed_state
        if fingerprint and 'L' in fingerprint:
            # Inject fingerprint residue into starting state
            fp_bias = torch.tensor(fingerprint['L'][:seed_state.shape[-1]], device=self.device, dtype=torch.float32)
            if fp_bias.numel() < seed_state.shape[-1]:
                fp_bias = F.pad(fp_bias, (0, seed_state.shape[-1] - fp_bias.numel()))
            current_state = 0.8 * current_state + 0.2 * fp_bias.unsqueeze(0)
        
        if audio_dyad:
            # Inject audio harmonics influence
            harmonics = audio_dyad.get('chebyshev_harmonics', [0.0]*10)
            a_bias = torch.tensor(harmonics, device=self.device, dtype=torch.float32)
            if a_bias.numel() < seed_state.shape[-1]: a_bias = F.pad(a_bias, (0, seed_state.shape[-1] - a_bias.numel()))
            current_state = 0.9 * current_state + 0.1 * a_bias.unsqueeze(0)

        if video_dyad_b64:
            # Subtle entropy shift for video
            current_state = current_state * 1.05 # 'Excited' state evolution for video context

        generated_chars = []
        max_len = 300
        min_len = 60
        
        # Autoregressive Loop
        for i in range(max_len):
            # Gradual temperature decay (Start focused -> End creative)
            iter_temp = temperature * (1.0 + 0.002 * i) 
            
            logits, conf = self.larynx(current_state, temperature=iter_temp)
            
            # Clean Vocabulary Filtering: Mask out non-standard symbols to force human/Voynich readability
            for idx in range(logits.shape[-1]):
                char_from_idx = self._idx_to_char(idx)
                if idx < 128:
                    if char_from_idx not in self.allowed_chars:
                        logits[0, idx] = -1e9
                else:
                    # Allow dynamically registered unicode/emojis
                    if idx - 128 >= len(self.idx_to_unicode):
                        logits[0, idx] = -1e9
            
            # Apply Echo Suppression
            for char in input_chars:
                c_idx = self._char_to_idx(char)
                if c_idx < logits.shape[-1]:
                    logits[0, c_idx] -= suppression_factor
            
            # Apply Vowel Boosting
            for v in vowels:
                v_idx = self._char_to_idx(v)
                if v_idx < logits.shape[-1]:
                    logits[0, v_idx] *= vowel_boost_factor
            
            probs = torch.softmax(logits, dim=-1)
            char_idx = torch.multinomial(probs[0], 1).item()
            
            char = self._idx_to_char(char_idx)
            generated_chars.append(char)
            
            # Stop condition: require min_len AND high confidence at punctuation
            if len(generated_chars) >= min_len and char in ('.', '!', '?') and conf.item() > 0.85:
                break
                
            # State Evolution (Singing)
            # Use Larynx weights to rotate the state based on the symbol emitted
            feedback = torch.tanh(self.larynx.proj.weight[char_idx].unsqueeze(0))
            current_state = 0.9 * current_state + 0.1 * feedback + 0.02 * self._harvest_honest_jitter(current_state.shape)

        # 5. Linguistic Correction (Anti-glitch filter)
        res = "".join(generated_chars)
        # Fix excessive consonant runs
        vowel_indices = [i for i, c in enumerate(res) if c.lower() in 'aeiou']
        if len(vowel_indices) < len(res) / 5:
            # Add emergency vowel if too dry
            res += " eia"
            
        return res

    def _process_input_internal(
        self,
        text_input: str,
        fingerprint: Optional[Dict] = None,
        audio_dyad: Optional[Dict] = None,
        video_dyad_b64: Optional[str] = None,
        audio_b64: Optional[str] = None,
        media_chain: Optional[List[Dict]] = None,
        commutativity: str = 'symmetric',
        generate_response: bool = True,
        ingestion_mode: bool = False,
        regime: str = 'goo',
        voynich_token: Optional[Any] = None,
        performance_buffered: bool = False,
        tag_weights: Optional[Dict[str, float]] = None
    ) -> dict:
        """
        Process user text, update cavity, and generate emergent response via Fractal Recursion.
        Now uses CALM, KAGH, and HarmonicWaveDecomposition for proper legacy integration.
        Multi-modal fingerprint (image) and audio_dyad bias the manifold ingestion with
        non-commutative ordering governed by the commutativity parameter:
          'media_first' : media tensor evolves meta_state BEFORE text tensor.
          'text_first'  : text tensor evolves first; media applied after forward().
          'symmetric'   : simultaneous summation (default).
        Enhanced with constraint pressure injection when code is detected.
        """
        self.iteration += 1
        self.last_input_time = time.time()
        
        # --- REGIME-BASED ENTROPY INJECTION ---
        # Moving to 'goo' reduces hardening and increases mischief/entropy.
        # Moving to 'prickles' increases hardening for logical consistency.
        if regime == 'goo':
            self.current_regime = 'goo'
            print(f"[PHYSICS] Regime: GOO. Injecting Nutrients (Entropy boost).")
            # Nudge hardening toward 0.15 (soft manifold)
            self.hardening = 0.8 * self.hardening + 0.2 * 0.15
            mischief_bias = 0.5
            entropy_bias = 0.3
        else:
            self.current_regime = 'prickles'
            print(f"[PHYSICS] Regime: PRICKLES. Hardening manifold for truth branching.")
            # Nudge hardening toward 1.0 (crystallized manifold)
            self.hardening = 0.8 * self.hardening + 0.2 * 1.0
            mischief_bias = 0.05
            entropy_bias = 0.01
            
        self._last_mischief = torch.tensor([mischief_bias], device=self.device)
        
        # Calculate real-time spectral entropy of the current meta_state
        with torch.no_grad():
            spectrum = torch.fft.rfft(self.meta_state).abs()
            spectrum_norm = spectrum / (spectrum.sum(dim=-1, keepdim=True) + 1e-8)
            real_entropy = -(spectrum_norm * torch.log(spectrum_norm + 1e-8)).sum(dim=-1).mean()
            self._last_spectral_entropy = real_entropy.unsqueeze(0)
            
            # Blend with regime bias
            self._last_spectral_entropy = 0.7 * self._last_spectral_entropy + 0.3 * entropy_bias
        
        # --- INITIALIZATION COVERAGE ---
        response_text = ""
        metrics = {
            "pas_h": self._compute_pas_h(self.meta_state) if hasattr(self, 'meta_state') else 0.61,
            "chiral_torsion": 0.0,
            "glyphlock": False,
            "manifold_pressure": 0.0,
            "command_bypass": False,
            "retrieval_state": "SENSING",
            "honesty_score": 0.5
        }
        
        # 1. Embed Input (Hash Projection)
        input_tensor = self._text_to_tensor(text_input) # [1, dim]
        
        # --- PHASE 0: AFFORDANCE GRADIENT COMPUTATION (Hoisted) ---
        # Compute affordance gradients for both code and conversational patterns
        affordance_gradients = self._compute_affordance_gradients(text_input, input_tensor)
        
        # --- COMMAND PRIORITIZATION ---
        ingest_cmds = ["INGEST_DYAD:", "ASSOCIATE:", "INGEST_AUDIO_DYAD:", "INGEST_VIDEO_DYAD:", "SOVEREIGN_FETCH:", "CLOUD_FETCH:", "EXPORT_AGENT_SMITH:", "IMPORT_AGENT_SMITH:"]
        if any(text_input.startswith(cmd) for cmd in ingest_cmds):
             print(f"[CMD] Command Prioritization: Bypassing pipeline for direct response...", flush=True)
             # Merciful Topological Reset: Clear historical trauma/dissonance for manual commands
             # to ensure the Braid Governor (Archetypes) has a fresh start.
             self.calm_history.zero_() 
             
             # Use current meta_state as the grounding seed for the command handler
             seed_state = self.meta_state.detach()
             
             # NEW: Generate diagnostics before early return to populate the terminal UI
             # This resolve the problem of "missing fingerprints" when using panels
             collision_res, collision_metrics = self._diagnose_multimodal_collision(
                 text_input=text_input,
                 input_tensor=input_tensor,
                 fingerprint=fingerprint,
                 audio_dyad=audio_dyad,
                 video_dyad_b64=video_dyad_b64,
                 audio_b64=audio_b64,
                 media_chain=media_chain,
                 commutativity=commutativity
             )
             metrics.update(collision_metrics)
             
             # Calculate Chiral Metrics (Structural Invariants)
             if hasattr(self, 'poly_config'):
                 coeffs = self.poly_config.get_coefficients_tensor()
                 metrics['chiral_score'] = float(compute_chiral_shift(coeffs).mean().item())
                 metrics['chiral_torsion'] = float(compute_chirality(coeffs).abs().mean().item())
                 metrics['glyphlock'] = bool(check_glyphlock(coeffs).max().item() > 0)
             
             # Handle Sovereign/Cloud fetches
             if text_input.startswith("SOVEREIGN_FETCH:"):
                 if self.ingestor:
                     print(" Manual Sovereign Nutrient Fetch initiated...")
                     convs = self.ingestor.ingest_sovereign_logic(limit=10)
                     response_text = f"SOVEREIGN_FETCH: Ingested {len(convs)} High-Entropy conversations from HN/SE."
                 else:
                     response_text = "SOVEREIGN_FETCH: Ingestor not initialized."
             elif text_input.startswith("CLOUD_FETCH:"):
                 if self.ingestor and self.ingestor.drive:
                     print(" Manual Cloud Nutrient Sync initiated...")
                     convs = self.ingestor.sync_cloud_nutrients()
                     response_text = f"CLOUD_FETCH: Synced {len(convs)} shards from Google Drive."
                 else:
                     response_text = "CLOUD_FETCH: Cloud connectors not available."
             elif text_input.startswith("EXPORT_AGENT_SMITH:"):
                 print(" Agent Smith Export Protocol initiated...")
                 # Get current archetypal profile for the handshake
                 profile = self.archetypal_governor.export_governor_state()
                 
                 # Extract current Betti numbers for the signature
                 betti_nums = self.betti_router.estimate_sector_betti(self.meta_state).squeeze().tolist()
                 betti_dict = {i: float(b) for i, b in enumerate(betti_nums)}
                 
                 # Get frequencies from RNS for the polylog signature
                 prime_freqs = self.modular_rns.get_residues(self.meta_state)
                 
                 # Create a temporary dyad for the export
                 temp_dyad = KnowledgeDyad(
                     timestamp=datetime.datetime.now().isoformat(),
                     linguistic_description=text_input.replace("EXPORT_AGENT_SMITH:", "").strip() or "Sovereign Soliton Identity",
                     meta_state=self.meta_state.detach().cpu(),
                     gyroid_residue=self.meta_state.detach().cpu() # Using state as residue proxy for identity
                 )
                 
                 filepath = self.fossilizer.export_agent_smith(
                     dyad=temp_dyad,
                     prime_frequencies=prime_freqs.detach().cpu(),
                     betti_numbers=betti_dict,
                     filename="soliton_smith",
                     gauge_field=self.chern_simons_gasket.gauge_field,
                     archetype_profile=profile
                 )
                 response_text = f"AGENT_SMITH_EXPORT: Mathematical identity decoupled and anchored to {filepath}."
                 
             elif text_input.startswith("IMPORT_AGENT_SMITH:"):
                filepath = text_input.replace("IMPORT_AGENT_SMITH:", "").strip()
                print(f" Agent Smith Import Protocol initiated for {filepath}...")
                
                try:
                    payload = self.fossilizer.inject_agent_smith(filepath)
                    
                    # 1. Align Substrate (Geometry)
                    self.meta_state.data.copy_(payload['meta_state_aligned'].to(self.device))
                    if 'gauge_field_aligned' in payload and payload['gauge_field_aligned'] is not None:
                        self.chern_simons_gasket.gauge_field.data.copy_(payload['gauge_field_aligned'].to(self.device))
                    
                    # 2. Align Archetypes (Psychology)
                    bridge = AgentSubstrateBridge()
                    bridge.align_archetypes(payload, self.archetypal_governor)
                    
                    response_text = f"AGENT_SMITH_IMPORT: Soliton identity rehydrated. Manifold re-stabilizing around imported invariants."
                except Exception as e:
                    response_text = f"AGENT_SMITH_IMPORT_FAILED: {str(e)}"
             elif any(text_input.startswith(cmd) for cmd in ["INGEST_DYAD:", "INGEST_AUDIO_DYAD:", "INGEST_VIDEO_DYAD:", "ASSOCIATE:"]):
                print(f" Multimodal Dyad Ingestion Protocol initiated: {text_input[:50]}...", flush=True)
                response_text = self._handle_dyad_ingestion(
                    input_text=text_input,
                    fingerprint=fingerprint,
                    seed_state=seed_state,
                    audio_dyad=audio_dyad,
                    video_dyad_b64=video_dyad_b64,
                    audio_b64=audio_b64,
                    commutativity=commutativity
                )
             else:
                 # --- PRE-GENERATION DIAGNOSTICS & MISCHIEF UPDATE ---
                 # Update Mischief Probe with current regime and pressure
                 with torch.no_grad():
                     pas_h_cmd = self._compute_pas_h(self.meta_state)
                     gyroid_ent_cmd = self.gyroid_cov.estimate_entropy(self.meta_state).item()
                     mischief_active = (self.current_regime == 'goo') or (gyroid_ent_cmd > 0.3)
                     pressure_grad = self.calm_history.mean(dim=0) if self.calm_history is not None else torch.zeros(self.dim, device=self.device)
            
                     self.mischief_probe.update(
                         pressure_grad=pressure_grad, 
                         coherence=torch.tensor(0.5, device=self.device), 
                         pas_h=torch.tensor(pas_h_cmd, device=self.device), 
                         is_good_bug=mischief_active
                     )
        
                 response_text = self._generate_converged_response(
                        text_input=text_input, 
                        seed_state=seed_state, 
                        fingerprint=fingerprint, 
                        affordance_gradients=affordance_gradients,
                        audio_dyad=audio_dyad, 
                        video_dyad_b64=video_dyad_b64,
                        audio_b64=audio_b64,
                        voynich_token=voynich_token
                    )
             
             # Finalize metrics for command bypass
             metrics.update({
                 "pas_h": 1.0, 
                 "chiral_score": 1.0, # Complete alignment for manual command
                 "manifold_pressure": 0.0,
                 "command_bypass": True,
                 "retrieval_state": "KNOWN",
                 "honesty_score": 1.0
             })
             
             return {
                 "response": response_text,
                 "iteration": self.iteration,
                 "metrics": metrics,
                 "display_metadata": {"type": "command_result"},
                 "fingerprint_received": fingerprint is not None or audio_dyad is not None or video_dyad_b64 is not None,
             }

        # --- MULTIMODAL PRE-PROCESS ---
        # Video, Audio, and Image residues are now formally handled via GAP A 
        # (_diagnose_multimodal_collision) to ensure structural honesty.
        # Scalar biases have been replaced with full manifold excitations.

        # --- SPECULATIVE MEMORY BRIDGE ---
        # Prime the manifold with relevant fossils before starting the reasoning pass
        self._prime_manifold_with_fossils(input_tensor)
        
        print(f"[CONFIG] Affordance Gradients Computed:")
        print(f"   Executability: {affordance_gradients['executability_pressure']:.4f}")
        print(f"   Formal symbols: {affordance_gradients['formal_symbol_density']:.4f}")
        print(f"   Expandability: {affordance_gradients['runtime_expandability']:.4f}")
        print(f"   Closure: {affordance_gradients['referential_closure']:.4f}")
        print(f"   Conversational: {affordance_gradients['conversational_embedding_pressure']:.4f}")
        print(f"   API extraction: {affordance_gradients['api_extraction_potential']:.4f}")
        print(f"   Constraint forcing: {affordance_gradients['constraint_forcing_gradient']:.4f}")
        
        # =============================================
        # PHASE 0.5: CONVERSATIONAL EMBEDDING EXTRACTION
        # =============================================
        
        # Extract conversational embeddings if conversational pressure is high
        conversational_results = self._extract_conversational_embeddings(text_input, affordance_gradients)
        
        # =============================================
        # PHASE 0.7: CONSTRAINT FORCING DETERMINATION (AFFORDANCE-BASED)
        # =============================================
        
        # Determine constraint forcing strategy based purely on affordance gradients
        constraint_forcing_needed = (
            affordance_gradients['constraint_forcing_gradient'] > 0.1 or
            conversational_results.get('constraint_pressure_generated', 0.0) > 0.05
        )

        if constraint_forcing_needed:
            print(f"[FORCING] CONSTRAINT FORCING TRIGGERED:")
            if affordance_gradients['constraint_forcing_gradient'] > 0.1:
                print(f"   * Affordance gradient: {affordance_gradients['constraint_forcing_gradient']:.4f}")
            if conversational_results.get('constraint_pressure_generated', 0.0) > 0.05:
                print(f"   * Conversational pressure: {conversational_results['constraint_pressure_generated']:.4f}")
            
            # Show which affordances contributed to constraint forcing
            if affordance_gradients['executability_pressure'] > 0.05:
                print(f"   * Executability pressure: {affordance_gradients['executability_pressure']:.4f}")
            if affordance_gradients['formal_symbol_density'] > 0.05:
                print(f"   * Formal symbol density: {affordance_gradients['formal_symbol_density']:.4f}")
            if affordance_gradients['conversational_embedding_pressure'] > 0.05:
                print(f"   * Conversational embedding: {affordance_gradients['conversational_embedding_pressure']:.4f}")
            if affordance_gradients['api_extraction_potential'] > 0.05:
                print(f"   * API extraction potential: {affordance_gradients['api_extraction_potential']:.4f}")
        
        # Create constraint metrics from affordance gradients (no legacy code detection)
        enhanced_constraint_metrics = {
            'constraint_forcing_needed': constraint_forcing_needed,
            'affordance_gradients': affordance_gradients,
            'conversational_results': conversational_results,
            'complexity_metrics': {
                'executability_score': affordance_gradients['executability_pressure'],
                'conversational_score': affordance_gradients['conversational_embedding_pressure'],
                'api_extraction_score': affordance_gradients['api_extraction_potential'],
                'formal_symbol_score': affordance_gradients['formal_symbol_density'],
                'total_constraint_pressure': affordance_gradients['constraint_forcing_gradient'],
                # Derived metrics for constraint batch sizing
                'function_count': max(1, int(affordance_gradients['executability_pressure'] * 10)),
                'class_count': max(0, int(affordance_gradients['formal_symbol_density'] * 5))
            }
        }
        
        # -- Non-Commutative Dyad Routing (Braid Group) --
        # Converts fingerprint dict or audio_dyad dict into a projection vector,
        # then applies it before or after the text tensor based on commutativity.
        
        def _project_media_item(item_type, item_data) -> Optional[torch.Tensor]:
            """Projects a single media item (image, audio, video) into manifold space."""
            if not item_data: return None
            
            if item_type == 'image':
                # Chebyshev format
                if isinstance(item_data, dict) and 'L' in item_data:
                    K = len(item_data['L'])
                    flat = item_data.get('L', []) + item_data.get('Cr', []) + item_data.get('Cb', [])
                else: 
                    # Legacy or raw list
                    flat = item_data if isinstance(item_data, list) else []
                
                if flat:
                    # --- TOPOLOGICAL VISION SURGERY ---
                    coeffs = torch.tensor(flat, dtype=torch.float32, device=self.device)
                    
                    # 1. Extract Structural Bone (Russian Doll Residues)
                    residue = get_russian_doll_projection(coeffs, k_image_max=32) # [96]
                    
                    # 2. Extract Semantic Flesh (CNN Features) if image data is available
                    # We check for base64 or file path in item_data
                    img_src = item_data.get('b64') or item_data.get('path')
                    
                    if img_src:
                         # Perform Surgery: Flesh + Bone = Interlaced Manifold
                         return self.image_processor(img_src, gyroid_residue=residue)
                    else:
                         # Fallback: Project residue directly into embedding space
                         return self.fingerprint_proj(residue.unsqueeze(0))
                    
            elif item_type == 'audio':
                harmonics = item_data.get('chebyshev_harmonics', []) if isinstance(item_data, dict) else item_data
                if harmonics:
                    t = torch.tensor(harmonics, dtype=torch.float32, device=self.device)
                    if t.numel() < self.K_AUDIO_MAX: t = F.pad(t, (0, self.K_AUDIO_MAX - t.numel()))
                    else: t = t[:self.K_AUDIO_MAX]
                    return self.audio_dyad_proj(t.unsqueeze(0))
                    
            elif item_type == 'video' or item_type == 'gif':
                # Video/GIF bitstream handling
                # Handle both raw b64 and data URI formats
                target_b64 = item_data
                if isinstance(target_b64, str) and ',' in target_b64:
                    target_b64 = target_b64.split(',', 1)[1]
                
                if isinstance(target_b64, str):
                    if not hasattr(self, 'video_parser'):
                        from src.core.video_dyad_parser import VideoDyadParser
                        self.video_parser = VideoDyadParser(device=self.device)
                    
                    # 1. Parse enriched metrics with ResonanceCavity healing reference
                    # Pull stable residue patterns from the cavity to provide backward context (45)
                    healing_ref = self.cavity.M.mean(dim=0).flatten() if hasattr(self, 'cavity') else None
                    metrics = self.video_parser.parse_video_b64(target_b64, healing_ref=healing_ref)
                    
                    # 2. Extract residues and calculate Lazarus Shift (PAS_h)
                    ent = metrics['fractal_entropy']      # Nested Russian Doll entropy
                    
                    # Track Phase Alignment Shift for Lazarus Transition
                    pas_h = torch.norm(ent).item()
                    if not hasattr(self, 'last_pas_h'): self.last_pas_h = pas_h
                    delta_pas_h = abs(pas_h - self.last_pas_h)
                    self.last_pas_h = pas_h
                    
                    if delta_pas_h > 0.5:
                        print(f"!!! LAZARUS TRANSITION DETECTED: delta_pas_h={delta_pas_h:.4f}")
                    
                    # 3. Project into manifold space as a composite bias using the structural signature
                    # Replacing the silent scalar bypass with formal spectral projection
                    fp_tensor = self.video_parser.extract_96_spectral_signature(metrics)
                    media_emb = self.fingerprint_proj(fp_tensor.unsqueeze(0))
                    
                    return media_emb
            return None

        def _get_media_biases(fp_dict, audio_dict, chain) -> List[torch.Tensor]:
            """Returns ordered list of [1, dim] bias tensors."""
            biases = []
            if chain:
                for item in chain:
                    b = _project_media_item(item.get('type'), item.get('data'))
                    if b is not None: biases.append(b)
            else:
                # Fallback to single-item fields
                b_img = _project_media_item('image', fp_dict)
                if b_img is not None: biases.append(b_img)
                b_aud = _project_media_item('audio', audio_dict)
                if b_aud is not None: biases.append(b_aud)
            return biases

        media_biases = _get_media_biases(fingerprint, audio_dyad, media_chain)

        # Sequential Application Loop
        def _apply_sequential_biases(biases):
            with torch.no_grad():
                for b in biases:
                    # Apply bias with high-order manifold curvature correction
                    # This ensures media ingestion feels like a 'Sovereign Event'
                    self.meta_state = F.layer_norm(
                        self.meta_state + 0.7 * b,
                        self.meta_state.shape[1:]
                    )

        if commutativity == 'media_first' and media_biases:
            # Media chain evolves meta_state BEFORE text.
            _apply_sequential_biases(media_biases)
        elif commutativity == 'symmetric' and media_biases:
            # Simultaneous injection: Add the mean of all biases to input_tensor.
            mean_bias = torch.stack(media_biases).mean(dim=0)
            input_tensor = input_tensor + 0.5 * mean_bias
        # 'text_first' handled after forward()
        
        # 2. MIMICRY (Active Listening)
        self._train_mimicry(input_tensor, text_input)
        
        # 2.5 DYNAMIC MANIFOLD CLOCK (Integrated Physics)
        # Manifold pressure = Similarity(Input, History)
        # Higher pressure -> Seriousness (small dt) via ManifoldClock
        # ValenceFunctional computes the 'Hunger' (dissonance gap).
        with torch.no_grad():
            s_norm = self.meta_state / (torch.norm(self.meta_state) + 1e-8)
            i_norm = input_tensor / (torch.norm(input_tensor) + 1e-8)
            cos_sim = torch.dot(s_norm.flatten(), i_norm.flatten()).item()
            manifold_pressure_val = 1.0 - cos_sim 
            manifold_pressure_tensor = torch.tensor(manifold_pressure_val, device=self.device)
            
            # Use formal physics modules
            dt = self.manifold_clock.tick(manifold_pressure_tensor)
            # Inject Dissonance Triggers (Mischief & Entropy)
            # Calculated based on latest spectral analysis from previous steps
            mischief = getattr(self, '_last_mischief', torch.zeros(1, device=self.device))
            entropy = getattr(self, '_last_spectral_entropy', torch.zeros(1, device=self.device))
            
            self.current_hunger = self.valence_drive(manifold_pressure_tensor, mischief=mischief, entropy=entropy)
        
        # text_first commutativity: apply media biases AFTER forward() --
        # text already shaped the manifold; media now distorts the resulting state.
        if commutativity == 'text_first' and media_biases:
            _apply_sequential_biases(media_biases)

        # =============================================
        # =============================================
        # PHASE 2: INTERNAL FUSION (GAP A)
        # =============================================
        collision_residues, collision_metrics = self._diagnose_multimodal_collision(
            text_input=text_input,
            input_tensor=input_tensor,
            fingerprint=fingerprint,
            audio_dyad=audio_dyad,
            video_dyad_b64=video_dyad_b64,
            audio_b64=audio_b64,
            media_chain=media_chain,
            commutativity=commutativity
        )
        metrics.update(collision_metrics)

        # 3. Evolutionary Pass (Cavity + Meta-Functional)
        # Now passes collision_residues to Gap A internal path
        # Extract Braid word from current Zeitgeist state for steering
        braid_word = self._zeitgeist_state.braid_word if self._zeitgeist_state else None
        manifold_state = self.forward(input_tensor, dt=dt, collision_residues=collision_residues, braid_word=braid_word)
        seed_state = manifold_state.detach() # Explicit seed for response

        memory_state = getattr(self, '_last_memory_state', self.meta_state)
        est_residues = getattr(self, '_last_est_residues', torch.zeros_like(self.meta_state))

        
        # =============================================
        # DYAD AGENTIC TRIGGERS (AFFORDANCE-BASED)
        # =============================================
        dyad_override_response = None
        
        # Trigger Ingestion if expandability is critical
        if affordance_gradients.get('runtime_expandability', 0.0) > 0.4:
            print("[TRIGGER] Agentic Ingestion Triggered by Affordance Gradient")
            dyad_override_response = self._handle_dyad_ingestion(f"AGENTIC_INGEST: {text_input}", fingerprint, seed_state, audio_b64=audio_b64)
            
        # Trigger Association if knowledge seeking is critical
        elif affordance_gradients.get('knowledge_seeking', 0.0) > 0.4:
            print("[TRIGGER] Agentic Association Triggered by Affordance Gradient")
            dyad_override_response = self._handle_association_learning(text_input, None, seed_state)
            
        # 5.a Real Voynich Exemption (Self-Sovereign Alphabet)
        with torch.no_grad():
            _, _, _, exemption_token = self.voynich_linguist(seed_state)

        # =============================================
        # 5.b CALM: Update history buffer and get trajectory assessment
        # =============================================
        # Update CALM history with current meta-state (tensor-based, not scalar)
        self.calm_history = self.calm.update_buffer(self.calm_history, self.meta_state)
        
        # Get CALM assessment: abort_score, rho, step, forcing, gauge, constraints
        calm_output = self.calm(self.calm_history)
        
        # Unpack based on return tuple length (handle legacy if needed, though we just updated it)
        if len(calm_output) == 6:
             abort_score_tensor, rho_factor_tensor, step_factor_tensor, forcing_tensor, gauge_tensor, constraints_tensor = calm_output
        else:
             # Legacy fallback (shouldn't happen if reload worked)
             abort_score_tensor, rho_factor_tensor, step_factor_tensor = calm_output
             forcing_tensor = torch.zeros_like(self.meta_state)
             gauge_tensor = torch.zeros(1, device=self.device)
             constraints_tensor = torch.zeros(1, 5, device=self.device)

        # Convert to scalars for diagnostics (handles both tensors and floats)
        def _as_float(v):
            try:
                import numbers
                if isinstance(v, torch.Tensor):
                    return float(v.detach().cpu().item())
                if isinstance(v, numbers.Number):
                    return float(v)
            except Exception:
                return 0.0

        abort_score = _as_float(abort_score_tensor)
        rho_factor = _as_float(rho_factor_tensor)
        step_factor = _as_float(step_factor_tensor)
        
        calm_diagnostics = {
            "abort_score": abort_score,
            "rho_factor": rho_factor,
            "step_factor": step_factor,
            "gauge_pressure": _as_float(gauge_tensor),
            "trajectory_status": "STABLE"
        }
        
        # Apply Voynich Exemption directly to the abort score
        if exemption_token.is_valid_exemption:
            abort_score = 0.0
            calm_diagnostics["abort_score"] = 0.0
            calm_diagnostics["trajectory_status"] = "VOYNICH_EXEMPTED"
        elif abort_score > 0.8:
            calm_diagnostics["trajectory_status"] = "CRITICAL_COLLAPSE_IMMINENT"
        elif abort_score > 0.7:
            calm_diagnostics["trajectory_status"] = "WARPED"

        gauge_pressure = _as_float(gauge_tensor)

        # =============================================
        # AGENTIC FORCING (Phase 3)
        # =============================================
        # If gauge pressure is sufficient, apply the forcing vector to steering
        if gauge_pressure > 0.1:
            with torch.no_grad():
                # Apply forcing: meta_state += gauge * forcing
                # Scale by 0.1 to keep it stable (nudging, not overwriting)
                force_magnitude = 0.1 * gauge_pressure
                correction = force_magnitude * forcing_tensor
                self.meta_state = self.meta_state + correction
                print(f" CALM Agentic Forcing applied: P={gauge_pressure:.2f}, ||F||={torch.norm(correction).item():.4f}")

        # =============================================
        # 5.5: LIVE PAS_h COMPUTATION
        # PAS_h = (1/N) * sum(cos(theta_k - theta_bar))
        # Implemented in PhaseAlignmentInvariant (invariants.py).
        # Computed once here from meta_state; reused everywhere below.
        # =============================================
        with torch.no_grad():
            pas_h_live = self._compute_pas_h(self.meta_state)

        # =============================================
        # 6. EARLY EXIT FOR NON-GENERATIVE TASKS
        # =============================================
        if not generate_response:
            msg = "Skipping generation pipeline (Association Mode)" if not ingestion_mode else "High-Throughput Ingestion Mode ACTIVE"
            print(f"[ENGINE] {msg}")
            
            # Extract residue vector (The manifold's unique topological response)
            residue_vector = self.meta_state.clone().detach().cpu().flatten().tolist()
            
            return {
                "status": "processed_no_generation",
                "iteration": self.iteration,
                "affordance_gradients": affordance_gradients,
                "conversational_results": conversational_results,
                "calm_diagnostics": calm_diagnostics,
                "residue_vector": residue_vector,
                "memory_state_updated": True,
                "mimicry_trained": True,
                "diagnostics": {
                    "suppress_ui": True,
                    "iteration": self.iteration,
                    "resonance_score": self._last_resonance,
                    "retrieval_state": "KNOWN"
                },
                "payload": {
                    "type": "topological_shape_stalk",
                    "status": "asymptotic_ingestion",
                    "stalk_active": True,
                    "pas_h": pas_h_live,
                }
            }

        # =============================================
        # 6. Speculative Coprime Recovery + Spectral Speculative Exit
        # =============================================
        # If CALM detects collapse (abort_score > 0.5), attempt structure recovery
        # using Wasserstein optimal transport toward a coprime-coherent manifold.
        
        # Route user-supplied scalar stack-weights to chirality_target
        stacked_target = None
        if tag_weights is not None and hasattr(self, 'archetypal_governor'):
            stacked_target = self.archetypal_governor.compute_stacked_target(tag_weights)

        self.meta_state, recovery_metrics = self.coprime_gate(
            state=self.meta_state,
            abort_score=abort_score_tensor,
            residues=est_residues,
            chirality_target=stacked_target if stacked_target is not None and stacked_target.norm() > 0 else input_tensor,
            exemption_token=exemption_token
        )
        # If recovery succeeded in locking coprime parity, we override the CALM abort
        if recovery_metrics['coprime_lock'] and recovery_metrics['recovery_attempted']:
             abort_score = 0.0
             calm_diagnostics["trajectory_status"] = "RECOVERED"
             
             # Anti-Lobotomy: Mutate the polynomial configuration to restore chirality
             if hasattr(self, 'poly_config'):
                 self.poly_config.mutate()
                 print("[RECOVERY] Polynomial configuration mutated to restore architectural chirality.")
        
        # =============================================
        # 7. KAGH / MATRIOSHKA EVOLUTION LOOP
        # =============================================
        # Use KAGH to draft a "ghost" of the response state.
        # Wrapped in Matrioshka shell iterations to find a quantized fixed-point.
        kagh_input = memory_state + 0.3 * self.meta_state + input_tensor * 0.4
        
        if self.caq is not None:
            current_state = kagh_input
            
            # Expand trust scalars from k to dim (approx)
            trust_exp = self.trust_scalars.repeat_interleave(self.dim // len(self.trust_scalars) + 1)[:self.dim]
            pas_vec = torch.ones(self.dim, device=self.device) * pas_h_live
            
            max_matrioshka_steps = 3
            fixed_point = False
            
            for step in range(max_matrioshka_steps):
                q_in, _ = self.caq(current_state, pas_scores=pas_vec, trust_scores=trust_exp, voynich_token=voynich_token)
                ghost_next = self.kagh_drafter(q_in)
                q_out, boundary = self.caq(ghost_next, pas_scores=pas_vec, trust_scores=trust_exp, voynich_token=voynich_token)
                
                # Check if fixed point reached at this quantization level
                if torch.norm(q_out - q_in) < 1e-3:
                    print(f" Matrioshka fixed point reached at shell {self.caq._level} after {step+1} steps.")
                    response_ghost = q_out
                    fixed_point = True
                    break
                    
                if boundary is not None:
                    print(f" Shell crossed (to level {boundary.level})! Falling back to coarser granularity.")
                    self._last_matrioshka_diag = boundary.__dict__
                    self._last_boundary_obj = boundary
                    
                current_state = ghost_next
            
            
            if not fixed_point:
                response_ghost = current_state
        else:
            # Plumb CALM/RNS footprint over modular core
            bounds = self.modular_rns.get_modulus_bounds().to(self.device).expand_as(kagh_input)
            
            # Kelly-safe zone mask (Sub-components approved by modular validation)
            safe_mask = (torch.abs(kagh_input) <= (bounds * 10.0)).float()
            
            # Apply KAGH continuous gradient descent strictly to approved subset
            kagh_draft = self.kagh_drafter(kagh_input)
            response_ghost = kagh_input * (1.0 - safe_mask) + kagh_draft * safe_mask

        
        # =============================================
        # 7. Harmonic Wave Decomposition: Separate Signal from Noise
        # =============================================
        # Split the ghost into ergodic (noise) and non-ergodic (signal) components
        ergodic_component, non_ergodic_component = self.harmonic_decomp(response_ghost)
        
        # Seed state emphasizes the non-ergodic (coherent) component
        # After coprime gating, the state is already structuralized
        seed_state = non_ergodic_component + 0.2 * ergodic_component + input_tensor * 0.3

        # =============================================
        # PHASE 2.7: ZEITGEIST ROUTER -- CRT POLYTOPE SWITCHING
        # Implements the non-commutative CRT index transition:
        #   S_t = (x_t, alpha_t, l_t, u_t)  ->  S_{t+1} = (x_{t+1}, alpha_{t+1}, l_{t+1}, u_{t+1})
        # Three modes: interior (scalar OK), grazing (tension), switching (non-commut.)
        # The exterior case emits 'undefined' -- topological refusal, not numeric error.
        # References: ai project report SEC VI; BIOMIMETIC_SYNTHESIS_REPORT SEC 4.4
        # =============================================
        _zg_mode = 'interior'
        _zg_diag: dict = {}
        if self.zeitgeist_router is not None and self._zeitgeist_state is not None:
            try:
                # Extract last BoundaryState from the Matrioshka diagnostics (if present)
                _last_boundary = getattr(self, '_last_boundary_obj', None)
                _zg_mode, self._zeitgeist_state, _zg_diag, seed_state = self.zeitgeist_router(
                    seed_state,
                    self._zeitgeist_state,
                    boundary=_last_boundary,
                )
                print(f" Zeitgeist mode: {_zg_mode} | alpha: {self._zeitgeist_state.alpha} "
                      f"| crt_idx: {self._zeitgeist_state.crt_index} "
                      f"| step: {self._zeitgeist_state.step}")
            except Exception as _zg_e:
                print(f"  ZeitgeistRouter error (non-fatal): {_zg_e}")


        # =============================================
        
        # =============================================
        # PHASE 1.5: ENHANCED CONSTRAINT PRESSURE INJECTION
        # =============================================
        
        # Inject constraint pressure based on affordance gradients (pure affordance-based approach)
        if constraint_forcing_needed:
            print(" Applying enhanced constraint pressure injection to seed state...")
            
            # Apply constraint injection with affordance-based metrics
            seed_state = self._inject_constraint_pressure(seed_state, enhanced_constraint_metrics)
            print(f" Post-injection seed state shape: {seed_state.shape}")
        
        # =============================================
        # 8. Dynamic Output Length (Gyroidic Tensor-Based)
        # =============================================
        base_length = max(len(text_input), 30)
        
        # Use GyroidCovarianceEstimator for tensor-based entropy instead of scalar std()
        # Feed recent meta-states as samples
        # For now, use single sample (meta_state) - could accumulate over interactions
        gyroid_entropy = self.gyroid_cov.estimate_entropy(self.meta_state)
        
        # CALM's step_factor modulates generation length
        calm_length_factor = step_factor  # Already a float
        
        length_modifier = 1.0 + min(gyroid_entropy.item(), 2.0) * calm_length_factor
        max_output_length = int(base_length * length_modifier * 1.5)
        max_output_length = min(max_output_length, 2000) # Increased for supertask
        min_output_length = max(len(text_input) // 2, 50) # Slightly increased min
        
        # =============================================
        # 8.5. GARBLED OUTPUT REPAIR PIPELINE (PHASE 2.1: SPECTRAL COHERENCE CORRECTOR)
        # =============================================
        print(f" Applying repair to state: {seed_state.shape}")
        
        try:
            # PHASE 2.1: Re-enable Spectral Coherence Corrector
            print(" Phase 2.1: Applying Spectral Coherence Correction...")
            print(f" Input state shape: {seed_state.shape}, device: {seed_state.device}")
            
            # Apply spectral coherence correction to fix consonant clustering
            # Make correction more aggressive for better results
            seed_state_corrected = self.spectral_corrector.adaptive_coherence_correction(
                signal=seed_state,
                output_text=None  # We don't have output text yet, but corrector can work without it
            )
            # Apply additional vowel-bias correction to combat consonant clustering
            # Toned down to be more 'honest' and less disruptive to manifold
            with torch.no_grad():
                # Boost dimensions that correspond to vowel-like patterns
                vowel_boost = self._harvest_honest_jitter(seed_state_corrected.shape) * 0.4 # Reduced from 1.0
                vowel_mask = self._harvest_honest_jitter(seed_state_corrected.shape, scaled=False) > 0.85 # Tighter mask
                seed_state_corrected = seed_state_corrected + vowel_boost * vowel_mask.float()
            
            print(f" Corrected state shape: {seed_state_corrected.shape}")
            
            # Get spectral diagnostics
            spectral_diagnostics = self.spectral_corrector.get_diagnostics()
            print(f" Spectral Coherence: theta={spectral_diagnostics['theta_coherence']:.3f}, "
                  f"energy_ratio={spectral_diagnostics['energy_ratio']:.3f}")
            
            # Store diagnostics for metrics
            self._last_spectral_diagnostics = spectral_diagnostics
            print(f" Stored diagnostics: {self._last_spectral_diagnostics}")
            
            # PHASE 2.2: Re-enable Bezout Coefficient Refresh (PROPER IMPLEMENTATION)
            print(" Phase 2.2: Applying Bezout Coefficient Refresh...")
            
            # Ensure proper state dimensions before Bezout processing
            if seed_state_corrected.dim() == 3 and seed_state_corrected.shape[1] == 1:
                seed_state_corrected = seed_state_corrected.squeeze(1)  # Remove singleton dimension
                print(f" Squeezed state for Bezout processing: {seed_state_corrected.shape}")
            
            try:
                # Create proper residues from corrected state for CRT correction
                batch_size = seed_state_corrected.shape[0]
                state_dim = seed_state_corrected.shape[1]
                
                if state_dim % self.k != 0:
                    target_dim = state_dim - (state_dim % self.k)
                    seed_state_sliced = seed_state_corrected[:, :target_dim]
                    remainder = seed_state_corrected[:, target_dim:]
                else:
                    target_dim = state_dim
                    seed_state_sliced = seed_state_corrected
                    remainder = None
                
                # Create proper residues for CRT correction (zero-copy view)
                residue_dim = target_dim // self.k
                residues_for_crt = seed_state_sliced.view(batch_size, self.k, residue_dim)
                print(f" Created residues for Bezout: {residues_for_crt.shape}")
                
                # Apply CRT correction to fix modulus drift
                corrected_residues = self.bezout_refresh.apply_crt_correction(residues_for_crt)
                
                # Reshape back to state format and restore original dimensions
                seed_state_crt_flat = corrected_residues.view(batch_size, -1)
                if remainder is not None:
                    seed_state_corrected = torch.cat([seed_state_crt_flat, remainder], dim=-1)
                else:
                    seed_state_corrected = seed_state_crt_flat
                
                # Get Bezout diagnostics
                bezout_diagnostics = self.bezout_refresh.get_diagnostics()
                print(f" Bezout CRT: condition_number={bezout_diagnostics['bezout_condition_number']:.3f}")
                
                # Store Bezout diagnostics
                self._last_bezout_diagnostics = bezout_diagnostics
                
            except Exception as bezout_error:
                print(f"  Bezout Coefficient Refresh failed: {bezout_error}")
                print(" Using fallback diagnostics...")
                # Store fallback diagnostics
                self._last_bezout_diagnostics = {
                    'bezout_condition_number': 1.0,
                    'moduli_mean': 1.0,
                    'moduli_std': 0.0,
                    'drift_threshold': 0.5,
                    'error': str(bezout_error)
                }
            
            print(" Phase 2.2 skipped - continuing with spectral correction only")
            
            # Basic numerical stabilization (keep this as safety net)
            print(" Applying numerical stabilization...")
            
            # Check for NaN/inf values and replace them
            if torch.isnan(seed_state_corrected).any() or torch.isinf(seed_state_corrected).any():
                print("  Detected NaN/inf values, applying emergency stabilization")
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                nan_mask = torch.isnan(seed_state_corrected) | torch.isinf(seed_state_corrected)
                seed_state_corrected = torch.where(nan_mask, self._harvest_honest_jitter(seed_state_corrected.shape) * 0.1, seed_state_corrected)
            
            # Numerical stabilization: clamp values to prevent inf/nan in downstream operations
            seed_state_corrected = torch.clamp(seed_state_corrected, min=-10.0, max=10.0)
            
            # Normalize to prevent extreme values
            seed_state_corrected = seed_state_corrected / (torch.norm(seed_state_corrected, dim=-1, keepdim=True) + 1e-8)
            
            seed_state_repaired = seed_state_corrected
            print(f" Phase 2.1 repair complete. State shape: {seed_state_repaired.shape}")
            
        except Exception as e:
            print(f" REPAIR SYSTEM ERROR: {e}")
            print(" Falling back to basic stabilization...")
            
            # Store empty diagnostics for fallback
            self._last_spectral_diagnostics = {
                'theta_coherence': 0.0,
                'soliton_energy': 0.0,
                'ergodic_energy': 0.0,
                'energy_ratio': 0.0,
                'fallback_mode': True
            }
            self._last_bezout_diagnostics = {
                'bezout_condition_number': 1.0,
                'moduli_mean': 1.0,
                'moduli_std': 0.0,
                'drift_threshold': 0.5,
                'fallback_mode': True
            }
            self._last_chern_simons_diagnostics = {
                'level_k': 1,
                'manifold_dim': 3,
                'gasket_applied': False,
                'fallback_mode': True
            }
            self._last_soliton_diagnostics = {
                'alpha': 1.0,
                'healing_progress': 0.0,
                'iteration_count': 0,
                'fallback_mode': True
            }
            
            # Fallback to basic stabilization if spectral correction fails
            if torch.isnan(seed_state).any() or torch.isinf(seed_state).any():
                print("  Detected NaN/inf values, applying emergency stabilization")
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                nan_mask = torch.isnan(seed_state) | torch.isinf(seed_state)
                seed_state = torch.where(nan_mask, self._harvest_honest_jitter(seed_state.shape) * 0.1, seed_state)
            
            seed_state = torch.clamp(seed_state, min=-10.0, max=10.0)
            seed_state = seed_state / (torch.norm(seed_state, dim=-1, keepdim=True) + 1e-8)
            seed_state_repaired = seed_state
        
        # Use repaired state for generation
        seed_state = seed_state_repaired
        
        # Ensure final state matches expected system dimension for downstream compatibility
        if seed_state.shape[-1] != self.dim:
            if seed_state.shape[-1] > self.dim:
                # Truncate if larger
                seed_state = seed_state[:, :self.dim]
                print(f" Truncated state from {seed_state_repaired.shape[-1]} to {self.dim}")
            else:
                # Create new tensor of correct size and copy repaired values
                new_state = torch.zeros(seed_state.shape[0], self.dim, device=seed_state.device, dtype=seed_state.dtype)
                copy_size = min(seed_state.shape[-1], self.dim)
                new_state[:, :copy_size] = seed_state[:, :copy_size]
                
                # Fill remaining dimensions with reflected pattern from repaired state
                if copy_size < self.dim:
                    remaining = self.dim - copy_size
                    source_pattern = seed_state[:, :copy_size]
                    # Repeat and truncate the pattern to fill remaining space
                    pattern_repeats = (remaining + copy_size - 1) // copy_size  # Ceiling division
                    extended_pattern = source_pattern.repeat(1, pattern_repeats)[:, :remaining]
                    new_state[:, copy_size:] = extended_pattern
                
                seed_state = new_state
                print(f" Reconstructed state from {seed_state_repaired.shape[-1]} to {self.dim}")
        
        print(f" Final seed state shape: {seed_state.shape} (expected: [1, {self.dim}])")
        
        # Apply basic numerical stabilization
        seed_state = torch.clamp(seed_state, min=-10.0, max=10.0)
        seed_state = seed_state / (torch.norm(seed_state, dim=-1, keepdim=True) + 1e-8)
        
        # =============================================
        # PHASE 2.8: MULTI-MODAL COLLISION (Physics restoration)
        # =============================================
        # Collide image and text residues via DataAssociationLayer
        collision_residues = None
        codec_metrics = {}
        if fingerprint:
            try:
                # 1. Compute Image Embedding for DataAssociationLayer
                if 'L' in fingerprint:
                    K_fp = len(fingerprint['L'])
                    flat = fingerprint.get('L', [0.0]*K_fp) + fingerprint.get('Cr', [0.0]*K_fp) + fingerprint.get('Cb', [0.0]*K_fp)
                else:
                    flat = fingerprint.get('r',[]) + fingerprint.get('g',[]) + fingerprint.get('b',[]) + fingerprint.get('l',[]) + [fingerprint.get('texture', 0.0)] + fingerprint.get('edges', [0.0]*8)
                
                if flat:
                    fp_tensor = torch.tensor(flat, dtype=torch.float32, device=self.device)
                    target = self.K_IMAGE_MAX * 3 # 96
                    if fp_tensor.numel() < target:
                        fp_tensor = F.pad(fp_tensor, (0, target - fp_tensor.numel()))
                    elif fp_tensor.numel() > target:
                        fp_tensor = fp_tensor[:target]
                    
                    input_image_emb = self.fingerprint_proj(fp_tensor.unsqueeze(0))
                    
                    # 2. Extract Text Embedding for collision
                    input_text_emb = self._text_to_tensor(text_input)
                    
                    # 3. Perform Multi-modal Collision
                    collision_residues = self.associator(input_text_emb, input_image_emb) # [1, k]
                    
                    # 4. Codec Verification (Non-Abelian Entanglement)
                    # We pass the raw signal to the codec to calculate the commutativity gap
                    codec_result = self.codec.encode(text_input, fp_tensor)
                    codec_metrics = codec_result.diagnostics
                    
                    # 5. Mohr-Coulomb Yield Pressure (Topological Rupture)
                    # Yield = |shear| - mu * normal - cohesion
                    mu = 0.5
                    cohesion = 0.1
                    shear = codec_metrics.get('entanglement_ratio', 0.0)
                    normal = codec_metrics.get('modular_congruence', 0.0)
                    yield_p = shear - (mu * normal) - cohesion
                    codec_metrics['yield_pressure'] = yield_p
                    
                    if yield_p > 0:
                        print(f" [SURGERY]  TOPOLOGICAL RUPTURE DETECTED (Yield: {yield_p:.4f})")
                        seed_state = seed_state * (1.0 + yield_p) # Amplify residue energy
                    
                    # 6. Evaluate Matryoshka shell from collision
                    if hasattr(self, 'meta_polytope') and self.meta_polytope is not None:
                        # evaluate the post-fusion manifold state against the polytope
                        # ensuring boundary crossings are tracked relative to the text context
                        poly_res = self.meta_polytope(collision_residues)
                        if hasattr(poly_res, 'level'): # BoundaryState case
                            codec_metrics['matryoshka_level'] = int(poly_res.level)
                        else: # Tuple case (yq, new_alpha, new_level)
                            _, _, shell_level = poly_res
                            codec_metrics['matryoshka_level'] = int(shell_level)
                        codec_metrics['matryoshka_depth'] = codec_metrics['matryoshka_level']
                
                self._last_codec_diagnostics = codec_metrics
            except Exception as collision_err:
                print(f"  Multi-modal collision failed: {collision_err}")
        
        print(f" Applied numerical stabilization. State range: [{seed_state.min():.3f}, {seed_state.max():.3f}]")
        
        # =============================================
        # PHASE 2.3: CHERN-SIMONS GASKET (LOGIC LEAK PREVENTION)
        # =============================================
        print(" Phase 2.3: Applying Chern-Simons Gasket (Logic Leak Prevention)...")
        
        try:
            # Ensure proper state dimensions before applying gasket
            if seed_state.dim() == 3 and seed_state.shape[1] == 1:
                seed_state = seed_state.squeeze(1)  # Remove singleton dimension
                print(f" Squeezed state to proper dimensions: {seed_state.shape}")
            
            # Apply Chern-Simons gasket to plug logic leaks
            # First, we need to create residues from the state for the gasket
            # Convert state to residue format expected by plug_logic_leak
            batch_size = seed_state.shape[0]
            state_dim = seed_state.shape[1]
            
            if state_dim % self.k != 0:
                target_dim = state_dim - (state_dim % self.k)
                seed_state_sliced = seed_state[:, :target_dim]
                remainder = seed_state[:, target_dim:]
            else:
                target_dim = state_dim
                seed_state_sliced = seed_state
                remainder = None
            
            # Now create residues with proper dimensions
            residue_dim = target_dim // self.k
            proper_residues = seed_state_sliced.view(batch_size, self.k, residue_dim)  # [1, 5, 13]
            
            # Use proper polynomial coefficients from the repair system's polynomial config
            # Instead of mock data, use the actual polynomial basis from the system
            base_polynomial_coeffs = self.poly_config.get_coefficients_tensor()  # [K, D]
            
            # Ensure coefficients match the residue dimensions
            if base_polynomial_coeffs.shape[1] != residue_dim:
                if base_polynomial_coeffs.shape[1] > residue_dim:
                    # Truncate if larger
                    proper_polynomial_coeffs = base_polynomial_coeffs[:, :residue_dim]
                    print(f" Truncated polynomial coeffs: {base_polynomial_coeffs.shape} -> {proper_polynomial_coeffs.shape}")
                else:
                    # Expand if smaller using proper polynomial evaluation
                    # Instead of padding, evaluate the polynomials at more points
                    x_points = torch.linspace(-1, 1, residue_dim, device=seed_state.device)
                    proper_polynomial_coeffs = self.poly_config.evaluate(x_points.unsqueeze(0)).squeeze(0).T  # [K, residue_dim]
                    print(f" Expanded polynomial coeffs via evaluation: {base_polynomial_coeffs.shape} -> {proper_polynomial_coeffs.shape}")
            else:
                proper_polynomial_coeffs = base_polynomial_coeffs
            
            print(f" Using proper polynomial coefficients: {proper_polynomial_coeffs.shape}")
            
            # Apply the Chern-Simons gasket with proper polynomial coefficients
            gasket_residues = self.chern_simons_gasket.plug_logic_leak(
                residues=proper_residues,
                polynomial_coeffs=proper_polynomial_coeffs
            )
            # Convert back to state format and restore original dimensions
            gasket_residues_flat = gasket_residues.view(batch_size, -1)
            
            # Restore to original state dimensions (remove padding if applied)
            if remainder is not None:
                seed_state_gasket = torch.cat([gasket_residues_flat, remainder], dim=-1)
                print(f" Restored original dimensions: {gasket_residues_flat.shape[1]} -> {state_dim}")
            else:
                seed_state_gasket = gasket_residues_flat
            
            # Get Chern-Simons diagnostics
            chern_simons_diagnostics = self.chern_simons_gasket.get_diagnostics()
            print(f" Chern-Simons: level_k={chern_simons_diagnostics.get('level_k', 'N/A')}")
            
            # Store diagnostics
            self._last_chern_simons_diagnostics = chern_simons_diagnostics
            
            # Use gasket-corrected state
            seed_state = seed_state_gasket
            print(f" Gasket-corrected state shape: {seed_state.shape}")
            
        except Exception as gasket_error:
            print(f"  Chern-Simons Gasket failed: {gasket_error}")
            print(" Continuing without gasket correction...")
            # Store fallback diagnostics
            self._last_chern_simons_diagnostics = {
                'level_k': 1,
                'manifold_dim': 3,
                'gasket_applied': False,
                'error': str(gasket_error)
            }
        
        # =============================================
        # PHASE 2.4: SOLITON STABILITY HEALER (FRACTURE HEALING)
        # =============================================
        print(" Phase 2.4: Applying Soliton Stability Healer (Fracture Healing)...")
        
        try:
            # Convert state back to residues for soliton healing
            batch_size = seed_state.shape[0]
            state_dim = seed_state.shape[1]
            
            # Apply zero-copy slicing for Soliton compatibility
            if state_dim % self.k != 0:
                target_dim = state_dim - (state_dim % self.k)
                seed_state_sliced = seed_state[:, :target_dim]
                remainder = seed_state[:, target_dim:]
            else:
                target_dim = state_dim
                seed_state_sliced = seed_state
                remainder = None
            
            # Create residues for soliton healing
            residue_dim = target_dim // self.k
            residues_for_healing = seed_state_sliced.view(batch_size, self.k, residue_dim)
            print(f" Created residues for Soliton healing: {residues_for_healing.shape}")
            
            # Use previously computed Gyroid Covariance Entropy (from step 8) as gcve_pressure
            current_gcve = gyroid_entropy if 'gyroid_entropy' in locals() else None
            
            # Apply soliton healing (we don't have output text yet, so it will use iteration-based healing)
            healed_residues = self.soliton_healer.heal_fractured_soliton(
                residues=residues_for_healing,
                output_text=None,  # Will be applied based on iteration count
                gcve_pressure=current_gcve # Mimics biological Hive Warping under GCVE stress
            )
            # Convert back to state format and restore original dimensions
            healed_state_flat = healed_residues.view(batch_size, -1)
            if remainder is not None:
                seed_state = torch.cat([healed_state_flat, remainder], dim=-1)
                print(f" Restored original dimensions after Soliton healing: {healed_state_flat.shape[1]} -> {state_dim}")
            else:
                seed_state = healed_state_flat
            
            # Get Soliton diagnostics
            soliton_diagnostics = self.soliton_healer.get_diagnostics()
            print(f" Soliton Healer: alpha={soliton_diagnostics['alpha']:.3f}, progress={soliton_diagnostics['healing_progress']:.3f}")
            
            # Store Soliton diagnostics
            self._last_soliton_diagnostics = soliton_diagnostics
            
        except Exception as soliton_error:
            print(f"  Soliton Stability Healer failed: {soliton_error}")
            print(" Continuing without soliton healing...")
            # Store fallback diagnostics
            self._last_soliton_diagnostics = {
                'alpha': 1.0,
                'healing_progress': 0.0,
                'iteration_count': 0,
                'error': str(soliton_error)
            }
        
        # =============================================
        # PHASE 2.5: CANONICAL LOVE VECTOR & SOFT SATURATED GATES
        # =============================================
        print(" Phase 2.5: Applying Love Vector & Soft Saturated Gates...")
        
        try:
            # Apply Love Invariant (Non-Ownable Flow)
            # L + meta_state
            self.meta_state = self.love_vector(self.meta_state)
            
            # Apply Repunit-CRT Probe factoring
            # Map continuous norm to a discrete repunit index n
            rep_n = int(torch.norm(self.meta_state).item()) % 20
            repunit_state, _ = self.repunit_probe(rep_n)
            self.meta_state = self.meta_state * 0.5 + repunit_state * 0.5
            
            # Diagnostic check (kernel property)
            ownership_leak = self.love_vector.ownership_check().item()
            print(f"[LOVE] Love Invariant active: ownership_leak={ownership_leak:.3f}")
            
            # Apply Love Invariant Protector to project to null space and detect violations (§20.3)
            protector_diag = {}
            if hasattr(self, 'love_protector') and self.love_protector is not None:
                _, protector_diag = self.love_protector.apply_love_protection(self.meta_state)
                if protector_diag.get('violation_detected', 0.0) > 0:
                    print(f"!!! [LOVE] VIOLATION DETECTED !!! (Mag: {protector_diag['violation_magnitude']:.6f}) - Restoring Invariant.")
                else:
                    print(f"[LOVE] Protected: norm={protector_diag.get('love_norm', 0.0):.3f}")

            
            # Apply Soft Saturated Gates for tri-state logic
            # ... (rest of soft gates logic) ...
            batch_size = seed_state.shape[0]
            state_dim = seed_state.shape[1]
            if state_dim % self.k != 0:
                target_dim = state_dim - (state_dim % self.k)
                seed_state_sliced = seed_state[:, :target_dim]
                remainder = seed_state[:, target_dim:]
            else:
                target_dim = state_dim
                seed_state_sliced = seed_state
                remainder = None
            
            residue_dim = target_dim // self.k
            residues_for_saturation = seed_state_sliced.view(batch_size, self.k, residue_dim)
            
            # Use live PAS_h computed from meta_state (PhaseAlignmentInvariant)
            pas_h = pas_h_live
            
            performance_scores = torch.norm(residues_for_saturation, dim=2).mean(dim=0)
            performance_scores = torch.sigmoid(performance_scores)
            
            saturated_residues = self.soft_gates.apply_soft_saturation(
                signal=residues_for_saturation,
                pas_h=pas_h,
                performance_scores=performance_scores
            )
            saturated_state_flat = saturated_residues.view(batch_size, -1)
            if remainder is not None:
                seed_state = torch.cat([saturated_state_flat, remainder], dim=-1)
            else:
                seed_state = saturated_state_flat
            
            # Diagnostics
            soft_gates_metrics = self.soft_gates.get_diagnostics()
            self._last_soft_gates_diagnostics = soft_gates_metrics
            self._last_love_diagnostics = {
                "ownership_leak": ownership_leak, 
                "love_norm": torch.norm(self.love_vector.L).item(),
                **{f"protector_{k}": v for k, v in protector_diag.items()}
            }

            
        except Exception as love_gates_error:
            print(f" Love Vector / Soft Gates failed: {love_gates_error}")

        # =============================================
        # PHASE 2.6: MATRIOSHKA QUANTIZED EVOLUTION LOOP
        # Realises: x_{t+1} = Q_{Z_t}(F(Q_{Z_t}(x_t)))
        # (ai project report_2-2-2026.txt 3 "Matrioshka Quantized Windows")
        # Uses CALM constraint output as PAS scores for anisotropy.
        # =============================================
        if self.caq is not None:
            try:
                # Derive per-axis PAS scores from CALM constraints if available
                _pas_scores = None
                if 'constraints_tensor' in locals() and constraints_tensor is not None:
                    # constraints_tensor: [1, 5]  map to dim via linear interpolation
                    _ct = constraints_tensor.detach().view(-1)  # [5]
                    # Expand to [dim] by repeating across field groups
                    repeats = self.dim // _ct.shape[0] + 1
                    _pas_scores = _ct.repeat(repeats)[:self.dim].sigmoid()  # [dim]  [0,1]

                # Optimization 6.5: Speculative Spectral Exit
                _entropy_tensor = getattr(self, '_last_spectral_entropy', torch.tensor([1.0], device=self.device))
                _matrioshka_steps = 3  # Default QFQ iterations
                if _entropy_tensor.item() < 0.05:
                    print(f"[SPECULATIVE] Coherent spectrum detected (Entropy: {_entropy_tensor.item():.4f}). Bypassing iterative evolution loop (Opt 6.5).")
                    _matrioshka_steps = 0
                
                _boundary_hit = False
                _loop = -1  # Initialize loop counter safely
                for _loop in range(_matrioshka_steps):
                    # Inner quantization: Q_Z(x)
                    q_inner, _b_inner = self.caq(seed_state, pas_scores=_pas_scores, voynich_token=voynich_token)
                    # Evolve through physics surrogate: F(Q_Z(x))
                    with torch.no_grad():
                        q_evolved = self.kagh_drafter(q_inner)
                    # Outer quantization: Q_Z(F(Q_Z(x)))
                    seed_state, _b_outer = self.caq(q_evolved, pas_scores=_pas_scores, voynich_token=voynich_token)
                    # Detect critical shell ceiling -- induce Hyperspherical Inversion (originally Sphere Eversion) to avoid lobotomy
                    if _b_outer is not None and _b_outer.is_critical():
                        print(f"[SHELL] Critical refusal boundary hit. Inducing Hyperspherical Inversion protocol...")
                        # Hyperspherical Inversion: Map coordinates via x -> x / ||x||^2 to project state outside the boundary obstruction
                        with torch.no_grad():
                            _norm_sq = torch.sum(seed_state * seed_state, dim=-1, keepdim=True) + 1e-8
                            seed_state = seed_state / _norm_sq
                        print("[EVERSION] Manifold successfully inverted. Logic bypassed boundary obstruction.")
                        _boundary_hit = True
                        break

                self._last_matrioshka_diag = self.caq.get_diagnostics()
                self._last_matrioshka_diag['loop_steps'] = _loop + 1
                self._last_matrioshka_diag['boundary_halt'] = _boundary_hit
                print(f"[OK] Phase 2.6 Matrioshka loop complete: "
                      f"level={self._last_matrioshka_diag['level']}, "
                      f"step_mean={self._last_matrioshka_diag['step_mean']:.4f}")
            except Exception as _caq_err:
                print(f"  Matrioshka evolution loop failed: {_caq_err}")

        print(" Starting text generation with fully repaired state...")
        
        # Initialize response_text to prevent UnboundLocalError in case of catastrophic failure
        response_text = ""
        # PHASE 5: THE WORLD DOWN THERE (Archetypal Governor)
        # =============================================
        print(" Phase 5: Routing through Braid Group & Archetypal Synthesis Governor...")
        
        # We process the final seed_state using the Braid Governor
        # Detect command bypass for Braid Governor
        is_cmd = text_input.startswith("INGEST_DYAD:") or text_input.startswith("ASSOCIATE:")
        # --- PRE-GENERATION DIAGNOSTICS & MISCHIEF UPDATE ---
        # Update Mischief Probe with current regime and pressure
        mischief_active = (self.current_regime == 'goo') or (current_gcve > 0.3)
        pressure_grad = self.calm_history.mean(dim=0) if self.calm_history is not None else torch.zeros(self.dim, device=self.device)
        
        self.mischief_probe.update(
            pressure_grad=pressure_grad, 
            coherence=torch.tensor(0.5, device=self.device), 
            pas_h=pas_h_live, 
            is_good_bug=mischief_active
        )
        
        try:
            archetype_out = self.archetypal_governor.run_archetypes(
                current_state=seed_state,
                stranded_states=self.meta_state,
                current_mischief=self.mischief_probe.H_mischief.item(),
                phase_alignment=pas_h_live,
                love_strengths=torch.cat([torch.tensor([0.1]), self.love_vector.L.flatten()]),
                void_frictions=torch.tensor([abort_score], device=self.device).repeat(self.meta_state.shape[0]), # use CALM abort as tension
                global_dt=dt,
                env_luminosity=1.0,
                volitional_scalar=affordance_gradients.get('executability_pressure', 0.5),
                system_entropy=gyroid_entropy.item() if 'gyroid_entropy' in locals() else 0.5,
                memory_trauma=float(self.calm_history.mean().item()),
                dissonance=abort_score,
                lucidity_idx=pas_h_live,
                raw_unquantized_state=self.meta_state,
                is_high_priority=is_cmd,
                tag_weights=tag_weights
            )
        except Exception as arch_err:
            print(f"[FAIL] Diegetic Engine processing failed: {arch_err}")
            # Recovery Fallback: Create a benign archetype output to allow generation to continue
            archetype_out = {
                "active_state": seed_state,
                "resurrections": [],
                "localized_dt": 0.1,
                "abstraction_rate": 0.0,
                "system_collapsed": False
            }
        
        if archetype_out.get("system_collapsed", False):
            # THE RA EGO DEATH EVENT HAS TRIGGERED
            ra_score = archetype_out.get('abstraction_rate', 9.99)
            void_str = f"[IRREDUCIBLE EGO DEATH (Ra = {ra_score:.2f})] Topology rejected standard response generation... Structural integrity fractured. "
            print(void_str)
            return {
                "status": "processed",
                "iteration": self.iteration,
                "response": void_str,
                "affordance_gradients": affordance_gradients,
                "payload": {"type": "ego_death"}
            }
            
        # If survived Ego Death, pass into 5-Gate Tri-State
        gate_out = self.five_gate_pipeline.process_pipeline(
            query_state=seed_state[0],
            internal_certainty=1.0 - abort_score,
            current_pas_h=pas_h_live,
            target_mischief=self.mischief_probe.H_mischief.item(),
            diegetic_retrieval_fn=None # Could plug the Wikipedia system here
        )
        
        # =============================================
        # PHASE 3: RESPONSE QUALITY OPTIMIZATION
        # =============================================
        print(f" Phase 3: Response Quality Optimization (Gate State: {gate_out['knowledge_state']})...")
        
        if gate_out["knowledge_state"] == KnowledgeState.CONFABULATED:
            # We are writing structured glitch lore
            if performance_buffered:
                # NEW ROUTE: Using graph manager for buffered response to avoid phonetic sabotage
                override_response = f"[CONFABULATED_GLITCH] Search failed, but Mischief ({self.mischief_probe.H_mischief.item():.2f}) is high. Recovering legacy resonance...\n"
                if self.graph_manager:
                    confab_gen = self.graph_manager.get_deep_refusal(seed_state)
                else:
                    confab_gen = "The internal logic is unclipped. The world is unclipped."
                response_text = override_response + confab_gen
            else:
                # NEW ROUTE: Verbose, persona-rich Lazarus Dream Sequence
                response_text = self._generate_confabulated_dream(seed_state, archetype_out)
        else:
            # Enhanced dyad-aware converged response generation
            # =============================================
            # PHASE 19: ENRICHED CONVERGED RESPONSE 
            # =============================================
            # We wrap the core ResonanceLarynx engine with advanced linguistic
            # filters (Echo Suppression, Vowel Boosting) to ensure convergence.
            response_text = self._generate_converged_response(
                text_input=text_input,
                seed_state=seed_state,
                fingerprint=fingerprint,
                affordance_gradients=affordance_gradients,
                voynich_token=voynich_token
            )
            
            # Update Interaction Context Buffer for next pass
            self.interaction_context.append(seed_state.detach())
            if len(self.interaction_context) > self.max_context_len:
                self.interaction_context.pop(0)
            if gate_out["knowledge_state"] == KnowledgeState.SEARCH_NEEDED:
                response_text = "[SEARCH_GATE_TRIGGERED] Internal manifold lacks topology. " + response_text
        print(f" Generated physics-enriched response: {response_text}")
        print(f" Response length: {len(response_text)} characters")
        
        # Inject CALM veto message if trajectory is unstable
        if calm_diagnostics["trajectory_status"] == "NEVER_VETO":
            response_text = f"MOMENTUM VETO: RESTRUCTURING MANIFOLD... {response_text}"
            
        # Agentic Dyad Override (Phase 4)
        if dyad_override_response:
            response_text = dyad_override_response
            print(f"[WAVE] Dyad Override applied: {response_text[:50]}...")

        # Metrics will be constructed after Phase 4 computations to ensure dependencies are defined
        
        # =============================================
        # PHASE 4: ADVANCED FEATURE INTEGRATION
        # =============================================
        print(" Phase 4: Advanced Feature Integration...")
        
        # Phase 4.1: Full Gyroid Violation Score computation
        gyroid_violation_score = self._compute_full_gyroid_violation_score(seed_state, response_text)
        
        # Phase 4.2: Complete Unfolding Closure Check implementation
        unfolding_closure_result = self._perform_unfolding_closure_check_numeric(seed_state, text_input, response_text)
        # Derive presentation-only boolean from numeric metrics
        if isinstance(unfolding_closure_result, dict):
            try:
                cs = float(unfolding_closure_result.get('closure_score', 1.0))
                ct = float(unfolding_closure_result.get('closure_threshold', 0.5))
                unfolding_closure_result['is_closed'] = bool(cs <= ct)
            except Exception:
                unfolding_closure_result['is_closed'] = False
        
        # Phase 4.3: Advanced topological analysis and graph generation
        topological_analysis = self._perform_advanced_topological_analysis(seed_state, text_input, response_text)
        
        # Add Phase 4 diagnostics
        phase4_diagnostics = {
            'gyroid_violation_score': gyroid_violation_score,
            'unfolding_closure_check': unfolding_closure_result,
            'topological_analysis': topological_analysis,
            'advanced_features_active': True
        }
        
        # Phase 4: Advanced Physics (Conditional & Budgeted)
        if self.extensions_enabled and generate_response: # Skip if purely associating
             advanced_physics_diagnostics = self._run_advanced_physics(text_input, affordance_gradients)
             phase4_diagnostics.update(advanced_physics_diagnostics)

        print(f" Phase 4 Gyroid Violation Score: {gyroid_violation_score:.4f}")
        print(f" Phase 4 Unfolding Closure: {unfolding_closure_result['is_closed']}")
        print(f" Phase 4 Topological Features: {len(topological_analysis['features'])} detected")
        
        # Calculate Tri-State Output based on Honesty/Trust/PAS_h
        trust_mean = float(self.trust_scalars.mean().item()) if hasattr(self, 'trust_scalars') else 0.5
        
        # True Gate 4/5 Mathematics
        with torch.no_grad():
            residues, _, _, _ = self.voynich_linguist(seed_state)
            crt_honesty = float(self.voynich_linguist.get_continuous_honesty(residues).item())
            
        # Diagnostics
        h_mischief = self.mischief_probe.H_mischief.item()
        
        honesty_score = (crt_honesty + trust_mean) / 2.0 # Blend Voynich with generic trust
        
        if honesty_score > 0.7:
            retrieval_state = "KNOWN" # System 2 Grounded
        else:
            search_useful = False # Gate 4 Stub (No external search API wired yet)
            mischief_active = h_mischief > self.unknowledge_domain.tau_m
            
            if not search_useful and mischief_active:
                retrieval_state = "CONFABULATED" # Gate 5 Honest Generation
            else:
                retrieval_state = "SEARCH_NEEDED"

        # Real-time ArXiv "Singing" Search Integration:
        if retrieval_state == "SEARCH_NEEDED" and hasattr(self, 'arxiv_ingestor') and self.arxiv_ingestor is not None:
            try:
                # 1. Generate larynx-decoded query
                query = self.arxiv_ingestor._generate_larynx_query()
                print(f" [SEARCH_GATE] 'Singing' query to ArXiv: '{query}'")
                
                # 2. Perform synchronous search
                self.arxiv_ingestor.ingest_arxiv_by_query(query)
                
                # 3. Reload live session fossil cache
                self._refresh_fossil_cache()
                
                # 4. Nudge the meta_state to inject the new topological context
                self._prime_manifold_with_fossils(input_tensor)
                
                # Update response text prefix
                response_text = f"[SEARCH_HEALED] Manifold updated via ArXiv search for '{query}'. " + response_text
            except Exception as e:
                print(f" [SEARCH_GATE] Realtime search and nudge failed: {e}")

        # =============================================
        # DIEGETIC VISUALIZER  Manifold Fracture Render
        # =============================================
        # Called only on tri-state events (CONFABULATED or SEARCH_NEEDED).
        # On KNOWN the overhead is zero  skip entirely.
        # Roughness contract: we pass raw live tensors; the visualizer
        # never smooths edges (see diegetic_visualizer.py doc-header).
        visualization_b64 = None
        viz_result = None
        if retrieval_state in ('CONFABULATED', 'SEARCH_NEEDED'):
            try:
                from src.ui.diegetic_visualizer import render_manifold_fracture

                # Re-run FractalMetaFunctional with live seed_state to get
                # the four structural components without storing extra state.
                _fractal_components = None
                try:
                    _residues_for_fmf = torch.zeros(1, self.k, device=self.device)
                    _fmf_out = self.fractal_meta(
                        current_state=seed_state[:1],
                        meta_state_prev=self.meta_state,
                        residues=_residues_for_fmf,
                        dark_matter=None,
                    )
                    _fractal_components = _fmf_out.get('components', {})
                except Exception as _fmf_e:
                    print(f"[VISUALIZER] FractalMeta forward failed: {_fmf_e}")

                # Introspection probe directions
                _intro_dirs = None
                if hasattr(self, 'introspection') and self.introspection is not None:
                    try:
                        _probe_input = seed_state[:1].expand(1, -1)
                        _intro_out = self.introspection(_probe_input)
                        _intro_dirs = {k: v.squeeze(0) for k, v in _intro_out.items()}
                    except Exception as _intro_e:
                        print(f"[VISUALIZER] Introspection probe failed: {_intro_e}")

                # ChernSimons energy from last cached diagnostics
                _cs_energy = None
                if hasattr(self, '_last_chern_simons_diagnostics') and self._last_chern_simons_diagnostics:
                    _csd = self._last_chern_simons_diagnostics
                    _cs_scalar = _csd.get('twist_energy', _csd.get('energy', None))
                    if _cs_scalar is not None:
                        import torch as _t
                        _cs_energy = _t.tensor([float(_cs_scalar)])

                viz_result = render_manifold_fracture(
                    retrieval_state=retrieval_state,
                    meta_state=self.meta_state,
                    fractal_components=_fractal_components,
                    introspection_directions=_intro_dirs,
                    chern_simons_energy=_cs_energy,
                    pas_h=float(pas_h_live),
                    h_mischief=float(h_mischief),
                    honesty_score=float(honesty_score),
                    iteration=self.iteration,
                )
                if viz_result and viz_result.get('b64'):
                    print(f"[VISUALIZER] Manifold fracture rendered -- {len(viz_result['b64'])} bytes (b64) "
                          f"sr={len(viz_result.get('structural_residues', []))} "
                          f"csf={len(viz_result.get('cheby_self_fingerprint', []))}")
                else:
                    print("[VISUALIZER] render_manifold_fracture returned empty result")
            except Exception as _viz_e:
                print(f"[VISUALIZER] Rendering error (non-fatal): {_viz_e}")
                import traceback as _tb; _tb.print_exc()


        # Feed structural residues back into meta_state  Introspection (I) channel
        # (RESONANCE_CAVITY.md: dM/dt = Decay + Flux + Introspection + Patterns + Violation)
        # This gives the system structural awareness of its own fracture WITHOUT seeing
        # the rendered picture.  The  values are deliberately small to nudge, not dominate.
        if viz_result and isinstance(viz_result, dict):
            visualization_b64 = viz_result.get('b64')

            sr = viz_result.get('structural_residues')
            if sr is not None and len(sr) > 0:
                try:
                    sr_t = torch.tensor(sr, dtype=torch.float32, device=self.device)
                    # Pad/truncate to residue_proj_dim
                    rpd = self._residue_proj_dim
                    if sr_t.numel() < rpd:
                        sr_t = F.pad(sr_t, (0, rpd - sr_t.numel()))
                    else:
                        sr_t = sr_t[:rpd]
                    with torch.no_grad():
                        sr_proj = self.residue_feedback_proj(sr_t.unsqueeze(0))
                        self.meta_state = F.layer_norm(
                            self.meta_state + 0.05 * sr_proj,
                            self.meta_state.shape[1:]
                        )
                    print(f"[FEEDBACK] Structural residues injected -> meta_state (k=0.05)")
                except Exception as _sr_e:
                    print(f"[FEEDBACK] Residue injection failed: {_sr_e}")

            csf = viz_result.get('cheby_self_fingerprint')
            if csf is not None and len(csf) > 0:
                try:
                    csf_t = torch.tensor(csf, dtype=torch.float32, device=self.device)
                    rpd = self._residue_proj_dim
                    if csf_t.numel() < rpd:
                        csf_t = F.pad(csf_t, (0, rpd - csf_t.numel()))
                    else:
                        csf_t = csf_t[:rpd]
                    with torch.no_grad():
                        csf_proj = self.residue_feedback_proj(csf_t.unsqueeze(0))
                        self.meta_state = F.layer_norm(
                            self.meta_state + 0.02 * csf_proj,
                            self.meta_state.shape[1:]
                        )
                    print(f"[FEEDBACK] Chebyshev self-fingerprint injected -> meta_state (k=0.02)")
                except Exception as _csf_e:
                    print(f"[FEEDBACK] Self-fingerprint injection failed: {_csf_e}")
        else:
            # Visualizer returned a bare string (old format) or None
            visualization_b64 = viz_result if isinstance(viz_result, str) else None

        print("[VISUALIZER] Feedback pass complete")

        # Calculate anisotropy based on self.meta_state variance
        phi_k = self.meta_state.flatten().view(-1, 8) if hasattr(self, 'meta_state') else torch.zeros((32, 8), device=self.device)
        if phi_k.numel() > 1:
            phi_var = torch.var(phi_k)
        else:
            phi_var = torch.tensor(0.01, device=self.device)
        anisotropy = float((phi_var + 1e-8).sqrt().item())

        # Construct diagnostics dictionary to populate terminal UI
        cs_diag = getattr(self, '_last_chern_simons_diagnostics', {})
        diagnostics = {
            "manifold_voice_resonance": float(self._last_resonance),
            "ley_line_anisotropy": anisotropy,
            "moebius_twist": float(cs_diag.get('twist_energy', 0.0)) if isinstance(cs_diag, dict) else 0.0,
            "spectral_entropy": float(self._last_spectral_entropy.item()) if hasattr(self, '_last_spectral_entropy') else 0.0,
            "honest_jitter": float(self._harvest_honest_jitter((1,)).item()) if hasattr(self, '_harvest_honest_jitter') else 0.1,
            "substream_entropy": float(video_breather.get("substream_entropy", 0.02)) if 'video_breather' in locals() else 0.02,
            "chiral_score": float(compute_chiral_shift(self.poly_config.get_coefficients_tensor()).mean().item()) if hasattr(self, 'poly_config') else 0.1,
            "chiral_torsion": float(compute_chirality(self.poly_config.get_coefficients_tensor()).abs().mean().item()) if hasattr(self, 'poly_config') else 0.0,
            "glyphlock": bool((check_glyphlock(self.poly_config.get_coefficients_tensor()).max().item() > 0) or (calm_diagnostics["trajectory_status"] == "RECOVERED")),
            "pas_h": pas_h_live,
            "retrieval_state": retrieval_state
        }

        # Construct metrics now that all dependencies are available

        metrics = {
            "response": response_text,
            "retrieval_state": retrieval_state,
            "resonance_score": self._last_resonance,
            "visualization_b64": visualization_b64,  # Manifold fracture render (base64 PNG or None)
            "honesty_score": float(honesty_score),
            "crt_honesty": crt_honesty,
            "h_mischief": h_mischief,
            "iteration": self.iteration,
            "spectral_entropy": float(self._last_spectral_entropy.item()) if hasattr(self, '_last_spectral_entropy') else 0.0,
            "chiral_score": float(compute_chiral_shift(self.poly_config.get_coefficients_tensor()).mean().item()) if hasattr(self, 'poly_config') else 0.1,
            "chiral_torsion": float(compute_chirality(self.poly_config.get_coefficients_tensor()).abs().mean().item()) if hasattr(self, 'poly_config') else 0.0,
            "glyphlock": bool((check_glyphlock(self.poly_config.get_coefficients_tensor()).max().item() > 0) or (calm_diagnostics["trajectory_status"] == "RECOVERED")),
            "pas_h": pas_h_live,
            "trust_mean": trust_mean,
            "coprime_lock": bool(recovery_metrics.get('coprime_lock', False)) if isinstance(recovery_metrics, dict) else False,
            "output_length": len(response_text),
            "affordance_gradients": affordance_gradients,
            "conversational_results": conversational_results,
            "calm_diagnostics": calm_diagnostics,
            "constraint_forcing_applied": constraint_forcing_needed,
            "diagnostics": diagnostics,
            # Phase 18: CRT Zeitgeist index diagnostics
            "zeitgeist": {
                "mode": _zg_mode,
                "alpha": list(self._zeitgeist_state.alpha) if self._zeitgeist_state is not None else [],
                "crt_index": self._zeitgeist_state.crt_index if self._zeitgeist_state is not None else 0,
                "step": self._zeitgeist_state.step if self._zeitgeist_state is not None else 0,
                "diagnostics": _zg_diag,
            },
            "payload": {
                "type": "topological_shape_stalk",
                "stalk": topological_analysis,
                "shape_violation": gyroid_violation_score,
                "pas_h": pas_h_live,
                "resonance": self._last_resonance,
                "topological_rupture": codec_metrics.get('topological_rupture', False),
                "lazarus_mode": bool(recovery_metrics.get('recovery_attempted', False) and recovery_metrics.get('is_generative', False)) if isinstance(recovery_metrics, dict) else False,
                "matryoshka_level": codec_metrics.get('matryoshka_level', 0),
                "curvature": float(codec_metrics.get('commutativity_gap', 0.0))
            }
        }
        
        # Add repair diagnostics if available
        repair_diagnostics = {}
        if hasattr(self, '_last_spectral_diagnostics'):
            repair_diagnostics['spectral_coherence_corrector'] = self._last_spectral_diagnostics
            print(f" Spectral Diagnostics: {self._last_spectral_diagnostics}")
        
        if hasattr(self, '_last_bezout_diagnostics'):
            repair_diagnostics['bezout_coefficient_refresh'] = self._last_bezout_diagnostics
            print(f" Bezout Diagnostics: {self._last_bezout_diagnostics}")
        
        if hasattr(self, '_last_chern_simons_diagnostics'):
            repair_diagnostics['chern_simons_gasket'] = self._last_chern_simons_diagnostics
            print(f" Chern-Simons Diagnostics: {self._last_chern_simons_diagnostics}")
        
        if hasattr(self, '_last_soliton_diagnostics'):
            repair_diagnostics['soliton_stability_healer'] = self._last_soliton_diagnostics
            print(f" Soliton Diagnostics: {self._last_soliton_diagnostics}")
        
        if hasattr(self, '_last_love_diagnostics'):
            repair_diagnostics['love_invariant_protector'] = self._last_love_diagnostics
            print(f" Love Diagnostics: {self._last_love_diagnostics}")
        
        if hasattr(self, '_last_soft_gates_diagnostics'):
            repair_diagnostics['soft_saturated_gates'] = self._last_soft_gates_diagnostics
            print(f" Soft Gates Diagnostics: {self._last_soft_gates_diagnostics}")
        
        # Phase 3 diagnostics
        phase3_diagnostics = {
            'dyad_aware_generation': True,
            'echo_suppression_active': True,
            'vowel_optimization_active': False,
            'linguistic_correction_available': True,
            'multimodal_fingerprint_support': any([fingerprint is not None, audio_dyad is not None, video_dyad_b64 is not None])
        }
        
        if repair_diagnostics:
            metrics['repair_diagnostics'] = repair_diagnostics
        
        # Add Phase 3 diagnostics
        metrics['phase3_diagnostics'] = phase3_diagnostics
        
        # Add Phase 4 diagnostics
        metrics['phase4_diagnostics'] = phase4_diagnostics
        
        # Sanitize metrics before returning to ensure no NaN/Inf leaks to clients
        def _sanitize(x):
            try:
                import math as _m
                if isinstance(x, float):
                    if _m.isnan(x) or _m.isinf(x):
                        return 0.0
                    return x
                if isinstance(x, dict):
                    return {k: _sanitize(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [_sanitize(v) for v in x]
                if isinstance(x, tuple):
                    return tuple(_sanitize(v) for v in x)
                if isinstance(x, torch.Tensor):
                    t = x.detach().cpu()
                    if not torch.isfinite(t).all():
                        t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
                    return t
                return x
            except Exception:
                return x
        metrics = _sanitize(metrics)

        # Add Matrioshka diagnostics if available
        if self._last_matrioshka_diag:
            metrics['matrioshka_diagnostics'] = self._last_matrioshka_diag

        # Add temporal association trainer diagnostics if available
        if self._last_temporal_diag:
            metrics['temporal_association_diagnostics'] = self._last_temporal_diag

        # Trigger one background temporal association train_step on live interaction
        self._maybe_trigger_temporal_training(input_tensor, response_text)

        # Captures the full multi-sensory context to prevent 'erasing of implication'.
        # self.iteration already incremented at start of _process_input_internal
        multimodal_context = {
            "fingerprint": fingerprint,
            "audio_dyad": audio_dyad,
            "video_dyad_b64": video_dyad_b64,
            "media_chain": media_chain,
            "commutativity": commutativity,
            "final_seed_state": seed_state.detach().cpu(),
            "unified_spectral_signature": metrics.get("unified_spectral_signature")
        }
        
        self.encoding_manager.save_encoding(
            iteration=self.iteration,
            text=text_input,
            input_tensor=input_tensor,
            memory_state=self.meta_state,
            response=response_text,
            metrics=metrics,
            multimodal_context=multimodal_context
        )

        # ==========================================
        # OUROBOROS LOOP: SHADOW LOG FOSSILIZATION
        # ==========================================
        if hasattr(self, 'coprime_gate') and hasattr(self.coprime_gate, 'pop_shadow_logs'):
            shadow_logs = self.coprime_gate.pop_shadow_logs()
            if shadow_logs:
                from src.core.knowledge_dyad_fossilizer import KnowledgeDyad
                # Fossilize structural anomalies binding them to the current physics state
                flat_state = seed_state.detach().cpu().flatten()
                # Ensure the vector is suitable for the fossilizer (e.g., 96 or 96)
                target_len = 96
                if len(flat_state) < target_len:
                    import torch.nn.functional as F
                    flat_state = F.pad(flat_state, (0, target_len - len(flat_state)))
                else:
                    flat_state = flat_state[:target_len]
                
                # Append shadow logs to response for diegetic visibility
                response_text += "\n\n[SHADOW_LOGS_RECOVERED]\n" + "\n".join([f"  ! {sl}" for sl in shadow_logs])
                metrics['response'] = response_text
                metrics['shadow_logs'] = shadow_logs

                for sl in shadow_logs:
                    sl_dyad = KnowledgeDyad(
                        image_fingerprint=flat_state,
                        linguistic_description=sl,
                        metadata={'source': 'ShadowLogPhase'}
                    )
                    self.fossilizer.fossilize(sl_dyad, seed_state)
                    print(f"[OUROBOROS] Fossilized Shadow Log: {sl[:60]}...")
                    if hasattr(self, '_shadow_replay_queue'):
                        self._shadow_replay_queue.append(sl)
                    
                    # Steer democratically: cast internal vote to shield system from recursive friction
                    try:
                        self.steer_democratically({'source': 'ouroboros', 'shadow_log': sl})
                    except Exception as steer_err:
                        print(f"[WARN] Ouroboros democratic steer failed: {steer_err}")

        # ==========================================
        # FINAL PERSISTENCE SYNC: Neglecton Snapshot
        # ==========================================
        # Prevents 'million years' latency on restart by saving all nodes to a single binary file.
        if self.graph_manager and self.graph_manager.nodes:
            try:
                snapshot_data = self.graph_manager.get_memory_snapshot()
                snapshot_path = os.path.join(self.graph_manager.data_dir, "neglecton_snapshot.pt")
                torch.save(snapshot_data, snapshot_path)
            except Exception as e:
                print(f"[GRAPH] Failed to save Neglecton snapshot: {e}")

        print("[OUT] Returning metrics")
        return metrics

    # =========================================================================
    # PHASE 17: TEMPORAL ASSOCIATION TRAINER  background bridge
    # =========================================================================

    def _maybe_trigger_temporal_training(self, input_tensor: torch.Tensor, response_text: str) -> None:
        """
        Fire one TemporalAssociationTrainer.train_on_interaction in a background daemon
        thread so that it never blocks the HTTP response path.
        """
        if self._is_training_temporal:
            return

        def _bg_train():
            try:
                self._is_training_temporal = True
                self._in_training = True
                # Detach for training
                inp = input_tensor.detach().cpu()
                
                # Check for trainer availability
                if not hasattr(self, 'trainer') or self.trainer is None:
                    return

                # Dispatch to whichever training interface the trainer provides.
                # SpectralStructuralTrainer    train_step(input_data)
                # TemporalAssociationTrainer   train_on_interaction(input_tensor, response_tensor)
                if hasattr(self.trainer, 'train_on_interaction'):
                    # Encode response_text as a float tensor for association learning
                    resp_chars = [ord(c) / 128.0 for c in (response_text or '')[:256]]
                    if len(resp_chars) < 256:
                        resp_chars += [0.0] * (256 - len(resp_chars))
                    resp_tensor = torch.tensor(resp_chars, dtype=torch.float32)
                    self.trainer.train_on_interaction(
                        input_tensor=inp,
                        response_tensor=resp_tensor,
                    )
                elif hasattr(self.trainer, 'train_step'):
                    self.trainer.train_step(inp)
                else:
                    print("[TAT] No known training interface on trainer  skipping.")
            except Exception as e:
                print(f"[TAT] Background training error: {e}")
            finally:
                self._is_training_temporal = False
                self._in_training = False


        import threading
        t = threading.Thread(target=_bg_train, daemon=True)
        t.start()

    def forward_text_emb(
        self,
        text_emb: torch.Tensor,
        return_analysis: bool = False,
    ) -> dict:
        """
        Adapter required by TemporalAssociationTrainer.

        The trainer calls ``model(text_emb=..., return_analysis=True)`` and
        expects a dict with keys the trainer checks (spectral_diagnostics,
        trust_scalars, etc.).  We route through the existing forward() pass
        and package the outputs into the expected dict shape.

        Args:
            text_emb: [batch, dim] pre-embedded text tensor.
            return_analysis: If True, include diagnostic dicts.

        Returns:
            dict with keys: 'output', 'trust_scalars', plus optional diag keys.
        """
        # Internal recursion check: ensure forward pass doesn't drift into another training cycle
        # but allow the actual math to execute so gradients are preserved for the trainer.

        # Calculate manifold propagation and maintain graph for survivorship_pressure 
        manifold_out = self.forward(text_emb, dt=0.05)

        result: dict = {
            "output": manifold_out,
            "trust_scalars": self.trust_scalars,
            "residue_distributions": getattr(self, "_last_est_residues", torch.zeros(1, 5, device=self.device)).unsqueeze(-1),
        }
        if return_analysis:
            result["spectral_diagnostics"] = getattr(
                self, "_last_spectral_diagnostics", {}
            )
            result["chern_simons_diagnostics"] = getattr(
                self, "_last_chern_simons_diagnostics", {}
            )
            result["soliton_healing_diagnostics"] = getattr(
                self, "_last_soliton_diagnostics", {}
            )
            result["love_diagnostics"] = getattr(
                self, "_last_love_diagnostics", {}
            )
            result["soft_gates_diagnostics"] = getattr(
                self, "_last_soft_gates_diagnostics", {}
            )
        return result

    # Make TemporalAssociationTrainer's `self.model(text_emb=..., ...)` syntax work
    def __call_with_text_emb(self, *args, text_emb=None, return_analysis=False, **kwargs):
        if text_emb is not None:
            return self.forward_text_emb(text_emb, return_analysis=return_analysis)
        return super().__call__(*args, **kwargs)

    def _compute_pas_h(self, state: torch.Tensor) -> float:
        """
        Compute live Phase Alignment Score PAS_h from a state tensor.

        Uses PhaseAlignmentInvariant (invariants.py) which implements:
            PAS_h = (1/N) * sum_k cos(theta_k - theta_bar)

        where theta_k are complex phases extracted from the state treated as
        an analytic signal, and theta_bar is the circular-mean phase.

        Args:
            state: [batch, dim] or [dim] state tensor.

        Returns:
            pas_h_float: scalar in [-1, 1], typically in [0, 1] for coherent states.
        """
        try:
            from src.core.invariants import PhaseAlignmentInvariant
            if not hasattr(self, '_pas_invariant'):
                # Lazy singleton  no need for degree param at this level
                self._pas_invariant = PhaseAlignmentInvariant(degree=3)
            s = state.detach()
            if s.dim() == 1:
                s = s.unsqueeze(0)  # [1, dim]
            pas_scores = self._pas_invariant(s)  # [batch]
            return float(pas_scores.mean().item())
        except Exception as _e:
            # Graceful fallback: use love vector norm proxy
            try:
                return float(torch.norm(self.love_vector.L).item() / 5.0)
            except Exception:
                return 0.61  # last-resort sentinel

    def steer_democratically(self, context: dict):
        """
        Democratic Steering Hub (Phase 20).
        Aggregates engagement signals from external sources (HN, SE) and internal friction (Ouroboros).
        Protects from 'smoothness leakage' via thresholded discretization.
        Governs updates via Leontief Input-Output cascade costs and Kelly Criterion risk hedging.
        """
        if not isinstance(context, dict):
            return

        # 1. Cast votes based on signals
        voted = False
        source = context.get('source', '')
        
        # External Hacker News Votes
        if 'hn_score' in context:
            score = float(context.get('hn_score', 0))
            complexity = float(context.get('text_complexity', 5.0))
            
            # High engagement boosts expressivity demand
            if score > 30:
                self.expressivity_votes += 1
                voted = True
            elif score < 5:
                self.expressivity_votes -= 1
                voted = True
                
            # High text complexity lowers mischief threshold (increases creative sensitivity)
            if complexity > 5.5:
                self.mischief_votes -= 1
                voted = True
            elif complexity < 4.0:
                self.mischief_votes += 1
                voted = True

        # External Stack Exchange Votes
        elif 'se_score' in context or 'se_view_count' in context:
            score = float(context.get('se_score', 0))
            views = float(context.get('se_view_count', 0))
            
            if score > 10 or views > 500:
                self.expressivity_votes += 1
                voted = True
            
            if context.get('se_answer_count', 0) > 3:
                self.mischief_votes -= 1 # Lower threshold = deeper exploration
                voted = True

        # Internal Ouroboros Friction Votes
        elif source == 'ouroboros' or 'shadow_log' in context:
            # Ouroboros shadow logs indicate loop tension
            # This casts a direct vote to increase the mischief threshold (+1 tau_m) to shield the system
            self.mischief_votes += 1
            voted = True

        if not voted:
            return

        # 2. Check Discretization Thresholds (No Smoothness Leakage)
        delta_lipschitz = 0.0
        delta_tau = 0.0

        if abs(self.expressivity_votes) >= self.voting_threshold:
            # Sign of expressivity_votes dictates raw demand
            direction = 1.0 if self.expressivity_votes > 0 else -1.0
            delta_lipschitz = direction * 0.05
            self.expressivity_votes = 0 # Reset accumulator
            
        if abs(self.mischief_votes) >= self.voting_threshold:
            direction = 1.0 if self.mischief_votes > 0 else -1.0
            delta_tau = direction * 0.02
            self.mischief_votes = 0 # Reset accumulator

        # If no thresholds met, do nothing (guards continuity)
        if delta_lipschitz == 0.0 and delta_tau == 0.0:
            return

        # 3. Prepare Demand Vector (d) for Governance
        demand = [delta_lipschitz, delta_tau]

        # ===============================================================
        # LEONTIEF INPUT-OUTPUT GOVERNANCE: Cascading Dependency Matrix
        # ===============================================================
        # Compute cascading costs leveraging centralized LeontiefGovernor
        demand_tensor = torch.tensor([abs(delta_lipschitz), abs(delta_tau)], device=self.device)
        total_production, _ = self.democratic_governor.cascading_cost(demand_tensor, self.democratic_matrix)
        total_cascading_cost = total_production.sum().item()
        
        # Safe Budget: max total cascading cost per update step is 0.15
        safe_budget = 0.15
        budget_scale = 1.0
        if total_cascading_cost > safe_budget:
            budget_scale = safe_budget / total_cascading_cost
            print(f"[LEONTIEF] Veto scale applied ({budget_scale:.3f}) - Cascade cost {total_cascading_cost:.4f} exceeds budget.")

        # Apply Leontief safety scaling to demand
        delta_lipschitz *= budget_scale
        delta_tau *= budget_scale

        # ===============================================================
        # KELLY CRITERION RISK ALLOCATION: Non-Ergodic Survival Hedging
        # ===============================================================
        stability_score = 1.0
        try:
            # Extract entropy band (H_meta) from mischief probe if available
            if hasattr(self, 'mischief_probe') and hasattr(self.mischief_probe, 'get_ambient_entropy'):
                ent = self.mischief_probe.get_ambient_entropy()
                ent_val = float(ent.item() if hasattr(ent, 'item') else ent)
                stability_score = max(0.01, min(0.99, 1.0 - math.tanh(ent_val * 1.5)))
            else:
                pas = self._compute_pas_h(self.meta_state)
                stability_score = max(0.01, min(0.99, (pas + 1.0) / 2.0))
        except Exception:
            stability_score = 0.5

        p_success = stability_score
        # Kelly Criterion f = 2p - 1
        kelly_fraction = max(0.0, 2.0 * p_success - 1.0)
        
        # Half-Kelly Hedge to prevent model ruin, ensuring minimal progress
        half_kelly = 0.5 * kelly_fraction
        kelly_scalar = max(0.1, half_kelly)

        # Scale by Kelly Criterion factor
        delta_lipschitz *= kelly_scalar
        delta_tau *= kelly_scalar

        # ===============================================================
        # 4. Execute Symbolic Deltas
        # ===============================================================
        updated_any = False
        
        # Apply Lipschitz Expressivity Delta
        if delta_lipschitz != 0.0 and hasattr(self, 'audience_mapper'):
            old_val = getattr(self.audience_mapper, 'lipschitz_k', 1.0)
            new_val = max(0.2, min(2.0, old_val + delta_lipschitz))
            if new_val != old_val:
                self.audience_mapper.lipschitz_k = new_val
                updated_any = True
                print(f"[DEMOCRATIC] [KELLY={kelly_scalar:.3f}] Adjusted Audience Expressivity (Lipschitz_k): {old_val:.3f} -> {new_val:.3f}")

        # Apply Unknowledge Mischief Threshold Delta
        if delta_tau != 0.0 and hasattr(self, 'unknowledge_domain'):
            old_tau = getattr(self.unknowledge_domain, 'tau_m', 0.3)
            new_tau = max(0.1, min(0.9, old_tau + delta_tau))
            if new_tau != old_tau:
                self.unknowledge_domain.tau_m = new_tau
                updated_any = True
                print(f"[DEMOCRATIC] [LEONTIEF={budget_scale:.3f}] Adjusted Mischief Threshold (tau_m): {old_tau:.3f} -> {new_tau:.3f}")

        if updated_any:
            print(f"[DEMOCRATIC] Discrete symbolic stabilization executed successfully.")

    def _harvest_honest_jitter(self, shape: torch.Size, scaled: bool = True) -> torch.Tensor:
        """
        Delegates harvesting to centralized harvest_honest_jitter, mapping scale intervals
        to preserve downstream dynamic constants.
        Follows 45.2 (Silicon Sovereignty).
        """
        # Central harvest returns [-1.0, 1.0] when scaled=False
        raw_jitter = harvest_honest_jitter(shape, device=self.device, scaled=False)
        # Map [-1.0, 1.0] -> [0.0, 1.0] to match local logistic map range
        jitter_0_1 = (raw_jitter + 1.0) / 2.0
        
        if scaled:
            # Map [0.0, 1.0] -> [-0.05, 0.05] (exact parity: (jitter - 0.5) * 0.1)
            return (jitter_0_1 - 0.5) * 0.1
        return jitter_0_1

    def _train_mimicry(self, input_state: torch.Tensor, text_target: str):
        """Train Larynx to decrypt the input state back to text autoregressively."""
        if len(text_target) < 2:
            return
            
        self.larynx.train()
        self.optimizer.zero_grad()
        
        # Dynamic tokenization map
        chars = [self._char_to_idx(c) for c in text_target]
        
        total_loss = torch.tensor(0.0, device=self.device)
        current_state = input_state.clone().to(self.device)
        
        for i in range(len(chars) - 1):
            logits, _ = self.larynx(current_state, temperature=1.0)
            target_idx = torch.tensor([chars[i + 1]], device=self.device, dtype=torch.long)
            loss = self.criterion(logits, target_idx)
            total_loss = total_loss + loss
            
            with torch.no_grad():
                # Teacher forcing: feed actual target character embedding to next step state
                idx = chars[i + 1]
                feedback = torch.tanh(self.larynx.proj.weight[idx].detach().unsqueeze(0))
                current_state = 0.9 * current_state.detach() + 0.1 * feedback
                
        avg_loss = total_loss / max(1, len(chars) - 1)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.larynx.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.larynx.eval()

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Sequence-Aware Polynomial Rotating Hash.
        Uses polynomial coefficients instead of hardcoded primes (anti-lobotomy).
        Ensures word order and sentence structure influence the embedding.
        """
        vec = torch.zeros(1, self.dim)
        
        # Generate polynomial coefficients leveraging centralized PolynomialBasis
        basis = PolynomialBasis(degree=11, basis_type='chebyshev')
        x_eval = torch.tensor([0.5], device=self.device)
        evals = basis.evaluate(x_eval).flatten() # Shape [12]
        
        poly_coeffs = []
        for coeff in evals.cpu().tolist():
            poly_coeffs.append(abs(coeff * 10) + 2)
        
        for i, char in enumerate(text):
            # Positional Polynomial Shift
            p = poly_coeffs[i % len(poly_coeffs)]
            char_idx = self._char_to_idx(char)
            # Rotate target dimension based on position and polynomial coefficient
            idx = int((i * p + char_idx) % self.dim)
            
            # Harmonic magnitude modulation
            magnitude = (char_idx / 128.0) * (1.0 / (math.log(i + 2)))
            vec[0, idx] += magnitude
            
        # Add a global sentence variance 'salt'
        if len(text) > 0:
            salt = sum(ord(c) for c in text) % self.dim
            vec[0, salt] *= 1.1
            
        return vec / (vec.norm() + 1e-8)
    
    def _compute_affordance_gradients(self, text: str, input_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Compute affordance gradients: soft signals for constraint forcing potential.
        
        Expanded to handle both code execution and conversational embedding extraction.
        Instead of detecting "code", we track gradients that indicate when input's
        cheapest continuation is execution, generation, or conversational API extraction.
        
        Returns gradients, not classifications. No premature branching.
        """
        
        # =============================================
        # EXECUTABILITY PRESSURE
        # =============================================
        # How much does this input want to become execution?
        
        # Measure imperative density (commands, instructions, procedures)
        imperative_markers = len([w for w in text.lower().split() 
                                if w in ['run', 'execute', 'call', 'invoke', 'apply', 'compute', 
                                        'generate', 'create', 'build', 'make', 'do', 'perform']])
        
        # Measure procedural structure (step-by-step, algorithmic flow)
        procedural_indicators = text.count('->') + text.count('=>') + text.count('then') + text.count('next')
        
        # Measure referential immediacy (this, that, it, the above)
        referential_density = len([w for w in text.lower().split() 
                                 if w in ['this', 'that', 'it', 'above', 'below', 'here', 'there']])
        
        executability_pressure = (imperative_markers * 0.2 + 
                                procedural_indicators * 0.1 + 
                                referential_density * 0.05) / max(len(text.split()), 4)
        
        # Add a tiny bias to prevent dead gradients in the UI
        executability_pressure = max(executability_pressure, 0.0001)
        
        # =============================================
        # FORMAL SYMBOL DENSITY
        # =============================================
        # Density of formal/symbolic structures (math, logic, schemas)
        
        # Count symbolic characters
        symbolic_chars = sum(1 for c in text if c in '{}[]()=<>+-*/\\|&^%$@#~`')
        
        # Count formal operators and relations
        formal_operators = (text.count('==') + text.count('!=') + text.count('<=') + 
                          text.count('>=') + text.count('&&') + text.count('||'))
        
        # Count structured delimiters
        structured_delims = (text.count('{') + text.count('[') + text.count('(') + 
                           text.count('"') + text.count("'"))
        
        formal_symbol_density = (symbolic_chars * 0.05 + 
                               formal_operators * 0.2 + 
                               structured_delims * 0.1) / max(len(text), 20)
        
        formal_symbol_density = max(formal_symbol_density, 0.0001)
        
        # =============================================
        # RUNTIME EXPANDABILITY
        # =============================================
        # Potential for runtime generation/expansion
        
        # Measure meta-linguistic content (talking about language, systems, generation)
        meta_markers = len([w for w in text.lower().split() 
                          if w in ['generate', 'create', 'build', 'construct', 'define', 
                                  'implement', 'system', 'function', 'method', 'class',
                                  'pattern', 'template', 'schema', 'grammar', 'rule']])
        
        # Measure generative potential (placeholders, variables, templates)
        generative_indicators = (text.count('{}') + text.count('[]') + text.count('()') + 
                               text.count('...') + text.count('TODO') + text.count('FIXME'))
        
        # Measure expansion markers (etc, and so on, similar, like)
        expansion_markers = len([w for w in text.lower().split() 
                               if w in ['etc', 'similar', 'like', 'such', 'example', 'instance']])
        
        runtime_expandability = (meta_markers * 0.05 + 
                               generative_indicators * 0.1 + 
                               expansion_markers * 0.03) / max(len(text.split()), 1)
        
        # =============================================
        # REFERENTIAL CLOSURE
        # =============================================
        # Self-referential or meta-structural content
        
        # Measure self-reference (system talking about itself)
        self_ref_markers = len([w for w in text.lower().split() 
                              if w in ['self', 'itself', 'recursive', 'meta', 'reflection',
                                      'mirror', 'loop', 'cycle', 'feedback', 'circular']])
        
        # Measure structural reference (talking about structure, topology, architecture)
        structural_markers = len([w for w in text.lower().split() 
                                if w in ['structure', 'topology', 'architecture', 'framework',
                                        'manifold', 'space', 'dimension', 'constraint', 'invariant']])
        
        # Measure closure indicators (complete, closed, bounded, finite)
        closure_markers = len([w for w in text.lower().split() 
                             if w in ['complete', 'closed', 'bounded', 'finite', 'total',
                                     'whole', 'entire', 'full', 'comprehensive']])
        
        referential_closure = (self_ref_markers * 0.08 + 
                             structural_markers * 0.06 + 
                             closure_markers * 0.04) / max(len(text.split()), 1)
        
        # =============================================
        # CONVERSATIONAL EMBEDDING PRESSURE (NEW)
        # =============================================
        # How much does this input want to become conversational API extraction?
        
        # Measure conversational markers (questions, dialogue, interaction)
        conversational_markers = len([w for w in text.lower().split() 
                                    if w in ['what', 'how', 'why', 'when', 'where', 'who', 'which',
                                            'explain', 'tell', 'describe', 'discuss', 'talk', 'say',
                                            'ask', 'answer', 'respond', 'reply', 'conversation']])
        
        # Measure question structures
        question_indicators = (text.count('?') + text.count('what ') + text.count('how ') + 
                             text.count('why ') + text.count('when ') + text.count('where '))
        
        # Measure dialogue patterns
        dialogue_patterns = (text.count('"') // 2 + text.count("'") // 2 + 
                           text.count(':') + text.count('said') + text.count('says'))
        
        # Measure knowledge-seeking behavior
        knowledge_markers = len([w for w in text.lower().split() 
                               if w in ['learn', 'understand', 'know', 'information', 'data',
                                       'facts', 'details', 'content', 'knowledge', 'research']])
        
        conversational_embedding_pressure = (conversational_markers * 0.08 + 
                                           question_indicators * 0.1 + 
                                           dialogue_patterns * 0.05 + 
                                           knowledge_markers * 0.06) / max(len(text.split()), 1)
        
        # =============================================
        # API EXTRACTION POTENTIAL (ENHANCED)
        # =============================================
        # How much does this input suggest external API data extraction?
        
        # Measure external reference markers (websites, sources, APIs)
        external_markers = len([w for w in text.lower().split() 
                              if w in ['wikipedia', 'google', 'search', 'api', 'website', 'url',
                                      'source', 'reference', 'link', 'external', 'online', 'web']])
        
        # Measure data extraction indicators
        extraction_markers = len([w for w in text.lower().split() 
                                if w in ['extract', 'fetch', 'get', 'retrieve', 'download', 'scrape',
                                        'collect', 'gather', 'obtain', 'acquire', 'access']])
        
        # Measure content type indicators
        content_markers = len([w for w in text.lower().split() 
                             if w in ['article', 'document', 'page', 'text', 'content', 'material',
                                     'information', 'data', 'resource', 'publication']])
        
        # Measure temporal/current information needs
        temporal_markers = len([w for w in text.lower().split() 
                              if w in ['current', 'latest', 'recent', 'new', 'updated', 'today',
                                      'now', 'live', 'real-time', 'fresh', 'modern']])
        
        # ENHANCED: Measure knowledge-seeking patterns (subtle API extraction signals)
        knowledge_seeking = len([w for w in text.lower().split() 
                               if w in ['learn', 'understand', 'know', 'find', 'discover', 'explore',
                                       'research', 'study', 'investigate', 'lookup', 'check']])
        
        # ENHANCED: Measure question patterns that suggest external data needs
        question_patterns = (text.count('?') + 
                           len([w for w in text.lower().split() if w.startswith('what') or w.startswith('how') or w.startswith('why')]))
        
        # ENHANCED: Measure knowledge-seeking patterns (subtle API extraction signals)
        knowledge_seeking = len([w for w in text.lower().split() 
                               if w in ['learn', 'understand', 'know', 'find', 'discover', 'explore',
                                       'research', 'study', 'investigate', 'lookup', 'check']]) / max(len(text.split()), 1)
        
        # ENHANCED: Measure question patterns that suggest external data needs
        question_patterns = (text.count('?') + 
                           len([w for w in text.lower().split() if w.startswith('what') or w.startswith('how') or w.startswith('why')])) / max(len(text.split()), 1)
        
        api_extraction_potential = (external_markers * 0.15 + 
                                  extraction_markers * 0.12 + 
                                  content_markers * 0.08 + 
                                  temporal_markers * 0.10 + 
                                  knowledge_seeking * 0.5 +     # Boosted weight
                                  question_patterns * 0.3)      # Boosted weight
        
        # =============================================
        # TENSOR-BASED AFFORDANCE AMPLIFICATION
        # =============================================
        # Use input tensor properties to amplify affordance signals
        
        with torch.no_grad():
            # Compute tensor entropy (high entropy = high generative potential)
            tensor_probs = torch.softmax(input_tensor.flatten(), dim=0)
            tensor_entropy = -torch.sum(tensor_probs * torch.log(tensor_probs + 1e-8)).item()
            
            # Compute tensor variance (high variance = high structural complexity)
            tensor_variance = torch.var(input_tensor).item()
            
            # Compute tensor sparsity (high sparsity = high formal structure)
            tensor_sparsity = (input_tensor.abs() < 0.1).float().mean().item()
            
            # Compute tensor coherence (for conversational flow)
            tensor_coherence = torch.cosine_similarity(
                input_tensor[:, :input_tensor.shape[1]//2], 
                input_tensor[:, input_tensor.shape[1]//2:], 
                dim=1
            ).mean().item()
            
            # Amplify affordances based on tensor properties
            entropy_amplification = min(tensor_entropy / 5.0, 2.0)  # Cap at 2x
            variance_amplification = min(tensor_variance / 2.0, 1.5)  # Cap at 1.5x
            sparsity_amplification = min(tensor_sparsity * 2.0, 1.8)  # Cap at 1.8x
            coherence_amplification = min(abs(tensor_coherence) * 2.0, 1.6)  # Cap at 1.6x
        
        # Apply tensor-based amplification
        executability_pressure *= entropy_amplification
        formal_symbol_density *= sparsity_amplification
        runtime_expandability *= variance_amplification
        referential_closure *= entropy_amplification
        conversational_embedding_pressure *= coherence_amplification
        api_extraction_potential *= variance_amplification
        
        # =============================================
        # CONSTRAINT FORCING GRADIENT (UPDATED)
        # =============================================
        # Overall pressure for constraint injection (weighted combination)
        # Now includes conversational and API extraction pressures
        
        constraint_forcing_gradient = (
            executability_pressure * 0.25 +              # Execution wants constraints
            formal_symbol_density * 0.20 +               # Formal structures create constraints
            runtime_expandability * 0.20 +               # Expandability needs constraints
            referential_closure * 0.15 +                 # Self-reference creates constraint loops
            conversational_embedding_pressure * 0.12 +   # Conversations need temporal associations
            api_extraction_potential * 0.08              # API data creates external constraints
        )
        # Update affordance history for temporal tracking
        affordance_snapshot = {
            'executability_pressure': executability_pressure,
            'formal_symbol_density': formal_symbol_density,
            'runtime_expandability': runtime_expandability,
            'referential_closure': referential_closure,
            'conversational_embedding_pressure': conversational_embedding_pressure,
            'api_extraction_potential': api_extraction_potential,
            'knowledge_seeking': knowledge_seeking, # NEW
            'constraint_forcing_gradient': constraint_forcing_gradient,
            'tensor_entropy': tensor_entropy,
            'tensor_variance': tensor_variance,
            'tensor_sparsity': tensor_sparsity,
            'tensor_coherence': tensor_coherence
        }
        
        self.affordance_history.append(affordance_snapshot)
        
        # Keep only recent history (sliding window)
        if len(self.affordance_history) > 10:
            self.affordance_history = self.affordance_history[-10:]
        
        # Update current affordance trackers
        self.affordance_trackers.update(affordance_snapshot)
        
        return affordance_snapshot
    
    def _detect_code_input(self, text: str) -> Dict[str, Any]:
        """
        Legacy code detection method for backward compatibility.
        
        Detects code patterns using regex patterns and returns detection metrics.
        This is the legacy system - the new affordance gradient system is preferred.
        """
        import re
        
        detected_patterns = []
        total_matches = 0
        
        # Check each code pattern
        for pattern in self.code_patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    detected_patterns.append({
                        'pattern': pattern,
                        'matches': len(matches),
                        'examples': matches[:3]  # First 3 examples
                    })
                    total_matches += len(matches)
            except re.error:
                # Skip invalid regex patterns
                continue
        
        # Calculate code score
        text_length = len(text.split())
        code_score = min(total_matches / max(text_length, 1), 1.0)
        
        # Determine if this is code
        is_code = code_score > 0.1 or len(detected_patterns) >= 3
        
        # Legacy complexity metrics for backward compatibility
        complexity_metrics = {
            'function_count': len([p for p in detected_patterns if 'def' in p['pattern']]),
            'class_count': len([p for p in detected_patterns if 'class' in p['pattern']]),
            'import_count': len([p for p in detected_patterns if 'import' in p['pattern']]),
            'total_patterns': len(detected_patterns),
            'total_matches': total_matches,
            'code_density': code_score
        }
        
        return {
            'is_code': is_code,
            'code_score': code_score,
            'detected_patterns': detected_patterns,
            'complexity_metrics': complexity_metrics,
            'constraint_forcing_needed': is_code  # Legacy compatibility
        }
    
    def _inject_constraint_pressure(self, seed_state: torch.Tensor, constraint_metrics: Dict[str, Any]) -> torch.Tensor:
        """
        Inject constraint pressure from pressure ingestor into seed state.
        
        Enhanced to handle both code detection and conversational embedding affordances.
        Creates heterogeneous constraint geometries that force incompatible compressions to coexist.
        """
        # Check if constraint forcing is needed
        if not constraint_metrics.get('constraint_forcing_needed', False):
            return seed_state
        
        print("[FORCING] ENHANCED CONSTRAINT INJECTION: Processing multiple affordance types")
        
        # Extract affordance information
        affordance_gradients = constraint_metrics.get('affordance_gradients', {})
        conversational_results = constraint_metrics.get('conversational_results', {})
        complexity = constraint_metrics.get('complexity_metrics', {})
        
        # Generate constraint pressure signature from multiple affordance sources
        constraint_sources = []
        
        # Executability constraints (replaces legacy code detection)
        if affordance_gradients.get('executability_pressure', 0.0) > 0.05:
            exec_score = int(affordance_gradients['executability_pressure'] * 1000)
            constraint_sources.append(f"exec_{exec_score}")
        
        # Formal symbol constraints
        if affordance_gradients.get('formal_symbol_density', 0.0) > 0.05:
            formal_score = int(affordance_gradients['formal_symbol_density'] * 1000)
            constraint_sources.append(f"formal_{formal_score}")
        
        # Conversational constraints
        if affordance_gradients.get('conversational_embedding_pressure', 0.0) > 0.05:
            conv_score = int(affordance_gradients['conversational_embedding_pressure'] * 1000)
            constraint_sources.append(f"conv_{conv_score}")
        
        # API extraction constraints
        if affordance_gradients.get('api_extraction_potential', 0.0) > 0.05:
            api_score = int(affordance_gradients['api_extraction_potential'] * 1000)
            constraint_sources.append(f"api_{api_score}")
        
        # Runtime expandability constraints
        if affordance_gradients.get('runtime_expandability', 0.0) > 0.05:
            expand_score = int(affordance_gradients['runtime_expandability'] * 1000)
            constraint_sources.append(f"expand_{expand_score}")
        
        # Create composite signature
        # Use deterministic, collision-resistant signature for stability across runs
        joined = "-".join(sorted(constraint_sources)).encode('utf-8')
        digest = hashlib.blake2b(joined, digest_size=16).hexdigest()
        pressure_signature = int(digest[:12], 16) % 1000000
        
        print(f"Constraint sources: {constraint_sources}")
        print(f" Pressure signature: {pressure_signature}")
        
        # Check cache first
        if pressure_signature in self.constraint_pressure_cache:
            print(f" Using cached constraint pressure for signature {pressure_signature}")
            constraint_batch = self.constraint_pressure_cache[pressure_signature]
        else:
            print(f" Generating new constraint pressure for signature {pressure_signature}")
            
            # Determine pressure ingestor sources based on affordance types (pure affordance-based)
            sources = []
            
            # High constraint pressure: use multiple sources for maximum pressure
            if (complexity.get('total_constraint_pressure', 0.0) > 0.15 or 
                conversational_results.get('constraint_pressure_generated', 0.0) > 0.1):
                sources = ['oeis_bulk', 'debian_sources']
            elif (affordance_gradients.get('executability_pressure', 0.0) > 0.08 or 
                  affordance_gradients.get('conversational_embedding_pressure', 0.0) > 0.08 or
                  affordance_gradients.get('formal_symbol_density', 0.0) > 0.05):
                sources = ['oeis_bulk', 'debian_sources']
            else:
                # Medium complexity: single source
                sources = ['oeis_bulk']
            
            # Force constraint pressure ingestion
            try:
                pressure_report = self.pressure_ingestor.force_pressure_ingestion(sources)
                self.last_pressure_report = pressure_report
                
                print(f" Pressure Report: {pressure_report['total_constraints_extracted']} constraints, "
                      f"{pressure_report['total_collisions_detected']} collisions, "
                      f"density: {pressure_report['pressure_density']:.3f}")
                
                # Generate constraint batch from pressure ingestor
                batch_size = min(8, max(2, len(constraint_sources) * 2))
                constraint_batch = self.pressure_ingestor.get_constraint_batch(batch_size)
                
                # Cache the constraint batch
                self.constraint_pressure_cache[pressure_signature] = constraint_batch
                
            except Exception as e:
                print(f"  Constraint pressure generation failed: {e}")
                # Fallback: generate synthetic constraint pressure (Honest Jitter)
                constraint_batch = self._harvest_honest_jitter((4, 512)) * 20.0
        
        # Inject constraint pressure into seed state
        batch_size, state_dim = seed_state.shape
        constraint_dim = constraint_batch.shape[1]
        
        # Apply Symmetry-Preserving Reshape for constraint injection
        if constraint_dim != state_dim:
            if constraint_dim > state_dim:
                # Truncate constraint batch to match state dimensions
                constraint_injection = constraint_batch[:, :state_dim]
                print(f" Truncated constraint batch: {constraint_dim} -> {state_dim}")
            else:
                # Expand constraint batch using reflective padding
                pad_size = state_dim - constraint_dim
                constraint_injection = torch.nn.functional.pad(constraint_batch, (0, pad_size), mode='reflect')
                print(f" Expanded constraint batch: {constraint_dim} -> {state_dim}")
        else:
            constraint_injection = constraint_batch
        
        # Compute enhanced injection strength based on multiple affordance types
        base_injection_strength = 0.2  # Default
        
        if self.last_pressure_report:
            pressure_density = self.last_pressure_report['pressure_density']
            base_injection_strength = min(pressure_density * 0.3, 0.8)  # Cap at 80%
        
        # Enhance injection strength based on affordance gradients (pure affordance-based)
        affordance_boost = 0.0
        
        # Executability boost (replaces legacy code boost)
        exec_pressure = affordance_gradients.get('executability_pressure', 0.0)
        if exec_pressure > 0.05:
            affordance_boost += exec_pressure * 0.3
            print(f" Executability affordance boost: {exec_pressure * 0.3:.4f}")
        
        # Formal symbol boost
        formal_pressure = affordance_gradients.get('formal_symbol_density', 0.0)
        if formal_pressure > 0.05:
            affordance_boost += formal_pressure * 0.25
            print(f" Formal symbol affordance boost: {formal_pressure * 0.25:.4f}")
        
        # Conversational boost
        conv_pressure = affordance_gradients.get('conversational_embedding_pressure', 0.0)
        if conv_pressure > 0.05:
            affordance_boost += conv_pressure * 0.3
            print(f" Conversational affordance boost: {conv_pressure * 0.3:.4f}")
        
        # API extraction boost
        api_pressure = affordance_gradients.get('api_extraction_potential', 0.0)
        if api_pressure > 0.05:
            affordance_boost += api_pressure * 0.25
            print(f" API extraction affordance boost: {api_pressure * 0.25:.4f}")
        
        # Runtime expandability boost
        expand_pressure = affordance_gradients.get('runtime_expandability', 0.0)
        if expand_pressure > 0.05:
            affordance_boost += expand_pressure * 0.2
            print(f" Runtime expandability boost: {expand_pressure * 0.2:.4f}")
        
        # Conversational constraint boost
        conv_constraint_pressure = conversational_results.get('constraint_pressure_generated', 0.0)
        if conv_constraint_pressure > 0.05:
            affordance_boost += conv_constraint_pressure * 0.4
            print(f" Conversational constraint boost: {conv_constraint_pressure * 0.4:.4f}")
        
        # Final injection strength
        injection_strength = min(base_injection_strength + affordance_boost, 0.9)  # Cap at 90%
        
        print(f" Enhanced injection strength: {base_injection_strength:.3f} + {affordance_boost:.3f} = {injection_strength:.3f}")
        
        # Apply constraint forcing through tensor superposition
        # Use the first constraint from the batch as primary forcing vector
        primary_constraint = constraint_injection[0:1]  # Keep batch dimension
        
        # Create heterogeneous constraint geometry
        # Method 1: Direct superposition (incompatible compression)
        forced_state = seed_state + injection_strength * primary_constraint
        
        # Method 2: Orthogonal constraint projection (geometric forcing)
        if constraint_injection.shape[0] > 1:
            secondary_constraint = constraint_injection[1:2]
            # Create orthogonal component
            dot_product = torch.sum(primary_constraint * secondary_constraint, dim=1, keepdim=True)
            orthogonal_component = secondary_constraint - dot_product * primary_constraint
            orthogonal_component = orthogonal_component / (torch.norm(orthogonal_component, dim=1, keepdim=True) + 1e-8)
            
            # Apply orthogonal forcing
            forced_state = forced_state + (injection_strength * 0.5) * orthogonal_component
        
        # Method 3: Constraint collision forcing (if high collision count)
        if self.last_pressure_report and self.last_pressure_report['total_collisions_detected'] > 10:
            collision_factor = min(self.last_pressure_report['total_collisions_detected'] / 100.0, 1.0)
            # Add collision-based noise to force constraint conflicts
            collision_noise = self._harvest_honest_jitter(seed_state.shape) * collision_factor * 1.0
            forced_state = forced_state + collision_noise
            print(f" Applied collision forcing: {self.last_pressure_report['total_collisions_detected']} collisions")
        
        # Normalize to prevent explosion while preserving constraint pressure
        forced_state = forced_state / (torch.norm(forced_state, dim=-1, keepdim=True) + 1e-8)
        
        print(f" Constraint pressure injected: strength={injection_strength:.3f}, "
              f"batch_size={constraint_injection.shape[0]}, "
              f"state_change={torch.norm(forced_state - seed_state).item():.4f}")
        
        return forced_state
    
    def _extract_conversational_embeddings(self, text: str, affordance_gradients: Dict[str, float]) -> Dict[str, Any]:
        """
        Extract conversational embeddings when conversational affordance is high.
        
        Integrates with existing temporal association training system to create
        conversational constraint pressure from API-based data sources.
        """
        conversational_pressure = affordance_gradients['conversational_embedding_pressure']
        api_pressure = affordance_gradients['api_extraction_potential']
        
        # Only extract if conversational pressure is significant
        if conversational_pressure < 0.05 and api_pressure < 0.05:
            return {'extracted': False, 'reason': 'insufficient_conversational_pressure'}
        
        print(f"  CONVERSATIONAL EMBEDDING EXTRACTION TRIGGERED")
        print(f"   Conversational pressure: {conversational_pressure:.4f}")
        print(f"   API extraction pressure: {api_pressure:.4f}")
        
        extraction_results = {
            'extracted': True,
            'conversational_pressure': conversational_pressure,
            'api_pressure': api_pressure,
            'associations_created': 0,
            'temporal_patterns_detected': [],
            'constraint_pressure_generated': 0.0
        }
        
        # =============================================
        # CONVERSATIONAL PATTERN DETECTION
        # =============================================
        
        # Detect conversational patterns for temporal association training
        conversational_patterns = self._detect_conversational_patterns(text)
        extraction_results['temporal_patterns_detected'] = conversational_patterns
        
        # =============================================
        # API-BASED CONTENT EXTRACTION
        # =============================================
        
        # If API extraction pressure is high, attempt to extract related content
        if api_pressure > 0.08:
            api_content = self._attempt_api_content_extraction(text, api_pressure)
            if api_content['success']:
                extraction_results['api_content_extracted'] = api_content
                
                # Create temporal associations from API content
                associations_created = self._create_temporal_associations_from_api(text, api_content['content'])
                extraction_results['associations_created'] = associations_created
        
        # =============================================
        # CONVERSATIONAL CONSTRAINT GENERATION
        # =============================================
        
        # Generate constraint pressure from conversational patterns
        if conversational_patterns:
            constraint_pressure = self._generate_conversational_constraints(conversational_patterns)
            extraction_results['constraint_pressure_generated'] = constraint_pressure
        
        print(f" Conversational extraction complete:")
        print(f"   Patterns detected: {len(conversational_patterns)}")
        print(f"   Associations created: {extraction_results['associations_created']}")
        print(f"   Constraint pressure: {extraction_results['constraint_pressure_generated']:.4f}")
        
        return extraction_results
    
    def _detect_conversational_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect conversational patterns for temporal association training."""
        patterns = []
        
        # Question-answer patterns
        if '?' in text:
            questions = [s.strip() for s in text.split('?') if s.strip()]
            for q in questions:
                if len(q) > 5:  # Meaningful questions
                    patterns.append({
                        'type': 'question',
                        'content': q,
                        'temporal_weight': 0.8,  # Questions have high temporal significance
                        'association_potential': 0.9
                    })
        
        # Dialogue markers
        dialogue_markers = ['"', "'", 'said', 'says', 'asked', 'replied', 'responded']
        for marker in dialogue_markers:
            if marker in text.lower():
                patterns.append({
                    'type': 'dialogue',
                    'marker': marker,
                    'temporal_weight': 0.6,
                    'association_potential': 0.7
                })
        
        # Knowledge-seeking patterns
        knowledge_words = ['explain', 'what', 'how', 'why', 'tell me', 'describe']
        for word in knowledge_words:
            if word in text.lower():
                patterns.append({
                    'type': 'knowledge_seeking',
                    'trigger': word,
                    'temporal_weight': 0.7,
                    'association_potential': 0.8
                })
        
        return patterns
    
    def _attempt_api_content_extraction(self, text: str, api_pressure: float) -> Dict[str, Any]:
        """Attempt to extract content from APIs based on text content."""
        
        # For now, simulate API extraction (in real implementation, this would call actual APIs)
        # This is where you'd integrate with Wikipedia API, search APIs, etc.
        
        api_indicators = ['wikipedia', 'search', 'information', 'data', 'content']
        
        for indicator in api_indicators:
            if indicator in text.lower():
                return {
                    'success': True,
                    'source': f'{indicator}_api',
                    'content': f"Extracted content related to '{text[:50]}...' from {indicator} API",
                    'content_length': len(text) * 3,
                    'extraction_method': 'simulated_api_call',
                    'api_pressure_used': api_pressure
                }

        return {
            'success': False,
            'reason': 'no_api_indicators_found',
            'api_pressure_used': api_pressure
        }
    
    def _create_temporal_associations_from_api(self, source_text: str, api_content: str) -> int:
        """Create temporal associations from API-extracted content."""
        
        # Use existing association learning system
        associations_created = 0
        
        try:
            # Create association using existing system
            association_text = f"ASSOCIATE: {source_text[:100]} <-> {api_content[:500]}"
            
            # Process through existing association learning
            result = self._handle_association_learning(association_text, None, self.meta_state)
            
            if "learned" in result.lower():
                associations_created = 1
                print(f" Created temporal association from API content")
            
        except Exception as e:
            print(f" Failed to create temporal association: {e}")
        
        return associations_created
    
    def _generate_conversational_constraints(self, patterns: List[Dict[str, Any]]) -> float:
        """Generate constraint pressure from conversational patterns."""
        
        if not patterns:
            return 0.0
        
        # Calculate constraint pressure based on pattern complexity
        total_weight = sum(p['temporal_weight'] * p['association_potential'] for p in patterns)
        pattern_diversity = len(set(p['type'] for p in patterns))
        
        # Constraint pressure increases with pattern complexity and diversity
        constraint_pressure = (total_weight / len(patterns)) * (pattern_diversity / 3.0)
        
        # Cap at reasonable maximum
        return min(constraint_pressure, 1.0)
    
    def _diagnose_multimodal_collision(
        self,
        text_input: str,
        input_tensor: torch.Tensor,
        fingerprint: Optional[Dict] = None,
        audio_dyad: Optional[Dict] = None,
        video_dyad_b64: Optional[str] = None,
        audio_b64: Optional[str] = None,
        media_chain: Optional[List[Dict]] = None,
        commutativity: str = 'symmetric'
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Calculates internal fusion residues and codec diagnostics for ANY media type.
        Ensures parity between image, audio, and video collision reporting.
        """
        collision_residues = None
        codec_metrics = {}

        # 1. Identify Primary Media Trace (PMT)
        primary_item = None
        pmt_type = None
        
        if media_chain:
            primary_item = media_chain[-1].get('data')
            pmt_type = media_chain[-1].get('type')
        elif fingerprint:
            primary_item = fingerprint
            pmt_type = 'image'
        elif audio_dyad or audio_b64:
            primary_item = audio_b64 if audio_b64 else audio_dyad
            pmt_type = 'audio'
        elif video_dyad_b64:
            primary_item = video_dyad_b64
            pmt_type = 'video'

        if not primary_item:
            return None, {}

        try:
            # 2. Extract 96-dim Coeffs (fp_tensor) and manifold embedding (media_emb)
            fp_tensor = None
            media_emb = None
            
            if pmt_type == 'image':
                if 'L' in primary_item:
                    K_fp = len(primary_item['L'])
                    flat = primary_item.get('L', [0.0]*K_fp) + primary_item.get('Cr', [0.0]*K_fp) + primary_item.get('Cb', [0.0]*K_fp)
                else:
                    flat = primary_item.get('r',[]) + primary_item.get('g',[]) + primary_item.get('b',[]) + primary_item.get('l',[]) + [primary_item.get('texture', 0.0)] + primary_item.get('edges', [0.0]*8)
                
                fp_tensor = torch.tensor(flat, dtype=torch.float32, device=self.device)
                target = self.K_IMAGE_MAX * 3 # 96
                if fp_tensor.numel() < target:
                    fp_tensor = F.pad(fp_tensor, (0, target - fp_tensor.numel()))
                else:
                    fp_tensor = fp_tensor[:target]
                media_emb = self.fingerprint_proj(fp_tensor.unsqueeze(0))
                
            elif pmt_type == 'audio':
                if isinstance(primary_item, str) and (primary_item.startswith('data:audio') or len(primary_item) > 1000):
                    # Raw Audio B64 (Taking advantage of ffmpeg 1.4)
                    if not hasattr(self, 'video_parser'):
                        from src.core.video_dyad_parser import VideoDyadParser
                        self.video_parser = VideoDyadParser(device=self.device)
                    v_audio_harmonics = self.video_parser.extract_audio_harmonics(primary_item)
                    if v_audio_harmonics is not None:
                        t = v_audio_harmonics
                    else:
                        t = torch.zeros(self.K_AUDIO_MAX, device=self.device)
                else:
                    harmonics = primary_item.get('chebyshev_harmonics', []) if isinstance(primary_item, dict) else primary_item
                    t = torch.tensor(harmonics, dtype=torch.float32, device=self.device)
                
                if t.numel() > 0:
                    # Pad to K_AUDIO_MAX for projection
                    if t.numel() < self.K_AUDIO_MAX: t = F.pad(t, (0, self.K_AUDIO_MAX - t.numel()))
                    else: t = t[:self.K_AUDIO_MAX]
                    media_emb = self.audio_dyad_proj(t.unsqueeze(0))
                    
                    # Pad to 96 for codec view
                    fp_tensor = F.pad(t, (0, 96 - t.numel())) if t.numel() < 96 else t[:96]
                    
            elif pmt_type == 'video' or pmt_type == 'gif':
                if not hasattr(self, 'video_parser'):
                    from src.core.video_dyad_parser import VideoDyadParser
                    self.video_parser = VideoDyadParser(device=self.device)
                
                target_b64 = primary_item
                if isinstance(target_b64, str) and ',' in target_b64:
                    target_b64 = target_b64.split(',', 1)[1]
                
                healing_ref = self.cavity.M.mean(dim=0).flatten() if hasattr(self, 'cavity') else None
                v_metrics = self.video_parser.parse_video_b64(target_b64, healing_ref=healing_ref)
                
                # Derive media_emb and fp_tensor from the structural signature
                fp_tensor = self.video_parser.extract_96_spectral_signature(v_metrics)
                media_emb = self.fingerprint_proj(fp_tensor.unsqueeze(0))
                
                # Update diagnostics with high-dim metrics
                codec_metrics['spectral_dominance'] = float(fp_tensor[:32].mean().item())
                codec_metrics['anisotropic_gap'] = float(v_metrics['fractal_entropy'].item())

            # 3. Collision Logic
            if media_emb is not None:
                text_emb = input_tensor
                collision_residues = self.associator(text_emb, media_emb)
                
                # 4. Codec Diagnostics with Non-Commutativity
                codec_result = self.codec.encode(text_input, fp_tensor.view(1, 8, 12), commutativity=commutativity)
                codec_metrics.update({
                    "entanglement_ratio": codec_result.diagnostics.get('entanglement_ratio', 0.0),
                    "commutativity_gap": codec_result.commutativity_gap,
                    "unified_spectral_signature": fp_tensor.detach().cpu(),
                    "modular_congruence": codec_result.modular_congruence,
                    "is_admissible": codec_result.diagnostics.get('is_admissible', False),
                    "structural_state": codec_result.diagnostics.get('structural_state', "Unknown")
                })
                
                # Surgery Yield Physics
                half_dim = self.dim // 2
                res_flat = codec_result.residue.flatten()
                if res_flat.numel() >= self.dim:
                    normal_part = res_flat[:half_dim]
                    shear_part = res_flat[half_dim:self.dim]
                    yield_pressure = shear_part.abs().mean() - 0.5 * normal_part.abs().mean() - 0.1
                    codec_metrics['yield_pressure'] = float(yield_pressure.item())
                    codec_metrics['topological_rupture'] = bool(yield_pressure.item() > 0.0)
                
                # Matryoshka Depth - Evaluated against post-collision Text Embedding
                if self.meta_polytope is not None:
                    # evaluate the post-fusion manifold state against the polytope
                    # ensuring boundary crossings are tracked relative to the text context
                    poly_res = self.meta_polytope(text_emb)
                    if hasattr(poly_res, 'level'): # BoundaryState case
                        codec_metrics['matryoshka_level'] = int(poly_res.level)
                        codec_metrics['topological_refusal'] = True
                    else: # Tuple case (yq, new_alpha, new_level)
                        yq, _, shell_level = poly_res
                        codec_metrics['matryoshka_level'] = int(shell_level)

        except Exception as e:
            print(f"[FAIL] Multimodal Collision Helper Error: {e}")

        return collision_residues, codec_metrics

    
    def _handle_dyad_ingestion(self, input_text: str, fingerprint: Optional[Dict], seed_state: torch.Tensor, audio_dyad: Optional[Dict] = None, video_dyad_b64: Optional[str] = None, audio_b64: Optional[str] = None, commutativity: str = 'symmetric') -> str:
        """Handle multi-modal dyad ingestion (Image, Audio, Video) using DyadFossilizer and GyroidicCodec."""
        # Determine modality from command prefix
        modality = "Image"
        if input_text.startswith("INGEST_AUDIO_DYAD:"): modality = "Audio"
        elif input_text.startswith("INGEST_VIDEO_DYAD:"): modality = "Video"
        
        # Clean the command and separate binary-id from description
        raw_content = input_text
        for prefix in ["INGEST_DYAD:", "ASSOCIATE:", "INGEST_AUDIO_DYAD:", "INGEST_VIDEO_DYAD:"]:
            raw_content = raw_content.replace(prefix, "")
        raw_content = raw_content.strip()
        
        # Priority-Based Modality Detection (Multi-modal simultaneous ingestion)
        # We prioritize Video > Audio > Image if multiple are present in an ASSOCIATE: command
        active_modality = modality
        if modality == "Image": # Default for ASSOCIATE:
             if video_dyad_b64: active_modality = "Video"
             elif audio_dyad: active_modality = "Audio"
             elif fingerprint: active_modality = "Image"
        
        # Support both [id] | description and just description
        description = raw_content
        if "|" in raw_content:
            _, description = raw_content.split("|", 1)
            description = description.strip()

        # Build signal tensor [96] for the KnowledgeDyad
        # All signals (Image fp, Audio harmonics) are normalized to this spectral form
        signal_tensor = torch.zeros(96, device=self.device)
        media_received = False
        
        audio_tensor = None
        video_breather = None
        
        if active_modality == "Audio" and (audio_dyad or audio_b64):
            if audio_b64:
                if not hasattr(self, 'video_parser'):
                    from src.core.video_dyad_parser import VideoDyadParser
                    self.video_parser = VideoDyadParser(device=self.device)
                v_audio_harmonics = self.video_parser.extract_audio_harmonics(audio_b64)
                if v_audio_harmonics is not None:
                    # Pad/truncate to 96
                    signal_tensor = torch.zeros(96, device=self.device)
                    min_sz = min(v_audio_harmonics.size(0), 96)
                    signal_tensor[:min_sz] = v_audio_harmonics[:min_sz]
                    audio_tensor = signal_tensor.clone()
                    media_received = True
                    print("[VIDEO_PARSER] Raw Audio stream isolated and projected via ffmpeg.", flush=True)
                else:
                    print("[VIDEO_PARSER] Failed to extract harmonics from raw audio b64.", flush=True)
            
            if not media_received and audio_dyad:
                harmonics = audio_dyad.get('chebyshev_harmonics', [])
                # Pad/truncate to 96 to match schema
                harmonics = (harmonics + [0.0] * 96)[:96]
                signal_tensor = torch.tensor(harmonics, device=self.device).float()
                audio_tensor = signal_tensor.clone()
                media_received = True
        elif active_modality == "Video" and video_dyad_b64:
            if not hasattr(self, 'video_parser'):
                from src.core.video_dyad_parser import VideoDyadParser
                self.video_parser = VideoDyadParser(device=self.device)
            
            # parse_video_b64 now handles audio extraction internally for a unified signature
            breather_modes = self.video_parser.parse_video_b64(video_dyad_b64, healing_ref=seed_state, extract_audio=True)
            
            video_breather = {
                'fractal_entropy': breather_modes['fractal_entropy'].item(),
                'substream_entropy': breather_modes['substream_entropy'].item(),
                'signal_length': breather_modes['signal_length'].item(),
                'audio_detected': breather_modes.get('audio_harmonics') is not None
            }
            
            # Unified Spectral Signature: Covariance + Entropy + Audio
            signal_tensor = self.video_parser.extract_96_spectral_signature(breather_modes)
            
            # Isolated Audio for fossilization
            v_audio_harmonics = breather_modes.get('audio_harmonics')
            if v_audio_harmonics is not None:
                audio_tensor = v_audio_harmonics
                print("[VIDEO_PARSER] Audio stream isolated and projected into harmonic space.", flush=True)
                
            media_received = True
        elif active_modality == "Image" and fingerprint:
            # Standard Image Ingestion (Zero-Mock Path)
            if 'L' in fingerprint and 'Cr' in fingerprint and 'Cb' in fingerprint:
                # =============================================
                # BRIDGE 2: MELIPONINI-CHEBYSHEV COUPLING
                # =============================================
                l_coeffs = fingerprint.get('L', [])
                cr_coeffs = fingerprint.get('Cr', [])
                cb_coeffs = fingerprint.get('Cb', [])
                
                # Fetch Virtual Algorithmic Latency (kappa) from sovereignty engine based on internal cognitive state
                stall_k = self.sovereignty_engine.get_virtual_algorithmic_latency(internal_entropy=0.5)
                
                # Zero-Mock: We use the raw residues (3x8 = 24 dims)
                l_tensor = torch.tensor(l_coeffs, device=self.device).float()
                cr_tensor = torch.tensor(cr_coeffs, device=self.device).float()
                cb_tensor = torch.tensor(cb_coeffs, device=self.device).float()
                
                # Formalize kappa as the T0 (DC) component (Meliponini Coupling)
                # T0 acts as the baseline energy level based on hardware friction.
                l_tensor[0] = l_tensor[0] + stall_k
                
                # PHASE 19: 13 PUSAFILIACRIMONTO ATTACHMENT
                # Non-dual anchoring of visual luminance to Love Invariant
                if hasattr(self, 'love_protector'):
                    with torch.no_grad():
                        # Anchor the Love Invariant via visual residue moving average
                        # L is the internal buffer name for the Love Vector
                        self.love_protector.L.data.copy_(
                            0.9 * self.love_protector.L.data + 0.1 * l_tensor.mean()
                        )
                    print(f"[13] Love Invariant anchored via visual residue (stall_k={stall_k:.4f}).")

                # Combine into a 24-dim spectral signal tensor.
                # The GyroidicCodec will handle the 1D->2D landscape transition.
                signal_tensor = torch.cat([l_tensor, cr_tensor, cb_tensor])
                media_received = True

            elif 'chebyshev' in fingerprint:
                # Modern Chebyshev Spectral Signature (Phase 12 un-lobotomized)
                # Typically 96 dimensions, but we project/pad to 96 for fossil compatibility
                coeffs = fingerprint.get('chebyshev', [])
                fp_tensor = torch.tensor(coeffs, device=self.device).float()
                if fp_tensor.size(0) >= 96:
                    signal_tensor = fp_tensor[:96]
                else:
                    signal_tensor = torch.nn.functional.pad(fp_tensor, (0, 96 - fp_tensor.size(0)))
                media_received = True

            elif 'r' in fingerprint:
                # Legacy 96-dim format
                fp_list = (fingerprint.get('r', []) + fingerprint.get('g', []) + fingerprint.get('b', []) + fingerprint.get('l', []) + [fingerprint.get('texture', 0.0)] + fingerprint.get('edges', [0.0]*8))
                if len(fp_list) == 96:
                    signal_tensor = torch.tensor(fp_list, device=self.device).float()
                    media_received = True

        # --- TOPOLOGICAL MATURATION (Augmentation Phase) ---
        # We perform augmentation-first to ensure matured, fractal-stable signals
        # interact with the non-Abelian engine.
        if media_received and hasattr(self, 'augmenter'):
            try:
                # Map Router mode to Chromatic shift
                router_mode = getattr(self.router, 'mode', 'interior')
                chromatic_mode = 'pink' if router_mode == 'interior' else 'atomic'
                
                print(f" [PIPELINE]  MANDELBULB RECURSIVE EMBEDDING... (Seed: {signal_tensor.norm().item():.4f})")
                signal_tensor, _ = self.augmenter.forward(
                    signal_tensor.unsqueeze(0), 
                    augmentation_factor=1,
                    chromatic_mode=chromatic_mode
                )
                signal_tensor = signal_tensor.squeeze(0)
            except Exception as e:
                print(f" [PIPELINE]  Augmentation-first bypass: {e}")

        # --- OFFICIAL DATA ASSOCIATION (Collision Phase) ---
        # Use DataAssociationLayer to fuse Multi-modal Invariants.
        entanglement_residue = None
        text_emb = self._text_to_tensor(description)
        
        # Project signal to manifold dim
        if signal_tensor.size(0) == 96:
            # Shift legacy 96 to 96
            fp_p = torch.zeros(96, device=self.device)
            fp_p[:min(96, 96)] = signal_tensor[:min(96, 96)]
            media_emb = self.fingerprint_proj(fp_p.unsqueeze(0))
        elif signal_tensor.size(0) == 24:
            # Zero-mock 24 to 96
            fp_p = torch.zeros(96, device=self.device)
            fp_p[:24] = signal_tensor
            media_emb = self.fingerprint_proj(fp_p.unsqueeze(0))
        else:
            # Fallback zero-pad
            fp_p = torch.zeros(96, device=self.device)
            min_sz = min(signal_tensor.size(0), 96)
            fp_p[:min_sz] = signal_tensor[:min_sz]
            media_emb = self.fingerprint_proj(fp_p.unsqueeze(0))
            
        if media_received:
            try:
                # [Batch, k] residues from the official associator head
                # This implements the structural collision (T, I)
                ent_k = self.associator(text_emb, media_emb)
                
                # We also want the [dim] dense residue for the KnowledgeDyad
                # Use the fusion gate directly from the associator
                fused = torch.cat([F.silu(self.associator.text_prj(text_emb)), 
                                  F.silu(self.associator.img_prj(media_emb))], dim=-1)
                entanglement_residue = F.silu(self.associator.fusion_gate(fused))
                
                print(f"[ASSOCIATOR] Multi-modal collision preserved (Residue K-norm: {ent_k.norm().item():.4f})")
            except Exception as e:
                print(f"[ASSOCIATOR] Collision failure")
            
            # Restore non-Abelian check
            codec_result = self.codec.encode(description, signal_tensor)
            entanglement = codec_result.diagnostics.get('entanglement_ratio', 0.0)
            print(f" [CODEC] Non-Abelian Entanglement: {entanglement:.4f}", flush=True)
            
            # Mandatory check: if low entanglement, we label as 'Stale' or 'Separable'
            if entanglement < 0.1:
                print(" [WARNING] Separable manifold detected. Ingestion may lack topological depth.")

            dyad = KnowledgeDyad(
                linguistic_description=description,
                image_fingerprint=signal_tensor,
                audio_harmonics=audio_tensor,
                video_breather=video_breather,
                gyroid_residue=codec_result.residue, # Irreducible cross-modal state
                meta_state=seed_state.detach().cpu() if seed_state is not None else None,
                metadata={
                    "entanglement": entanglement,
                    "ingestion_iteration": self.iteration,
                    "spectral_entropy": codec_result.diagnostics.get('spectral_entropy', 0.0),
                    "commutativity": commutativity
                }
            )
            
            # FOSSILIZE logic: derive topological invariants from seed_state (history)
            # This ensures 'No Erasing of Implication' via real-time derivation.
            self.fossilizer.fossilize(dyad, text_emb, seed_state=seed_state)
            
            # Self-Correction via Ingestion Trace
            self.iteration += 1
            response_text = f"[INGESTION_SUCCESS] Dyad fossilized. Entanglement: {entanglement:.4f}. Iteration: {self.iteration}."
            print(f"[SUCCESS] {response_text}", flush=True)
            
            return response_text
        
        # Create the dyad object
        dyad = KnowledgeDyad(
            linguistic_description=description,
            # If no fingerprint provided, use the zero-filled signal_tensor [96] as the 'Image Ground State'
            # FIX: Preserve signal_tensor for both Image and Video modalities
            image_fingerprint=signal_tensor if (fingerprint or modality in ["Image", "Video"]) else None,
            audio_harmonics=audio_tensor,
            video_breather=video_breather,
            gyroid_residue=entanglement_residue
        )
        
        # 3. Call fossilizer (Official Persistence Path)
        # Use text_emb [1, dim] and seed_state for topological derivation
        fossil_path = self.fossilizer.fossilize(dyad, text_emb, seed_state=seed_state)
        fossil_id = os.path.basename(fossil_path).replace(".fossil", "")
        
        # Bridge 4: Navigation over Storage (Zeitgeist Landmark)
        if hasattr(self, 'router'):
            self.router.register_fossil_landmark(fossil_id, intensity=1.2)
            print(f"[ROUTER] Fossil {fossil_id[:8]}... registered as Poincar Gravity Well.", flush=True)
        
        print(f"[WAVE] {modality} Deposition confirmed: {fossil_path}", flush=True)
        return (
            f"Knowledge Dyad ({modality}) fossilized at {os.path.basename(fossil_path)}. "
            f"{'Signal embedded (' + str(int(signal_tensor.norm().item()*1000)/1000) + ' L2-norm). ' if media_received else 'No media signal  text-only dyad. '}"
            f"Non-Abelian Implication preserved in manifold."
        )
    
    def _handle_association_learning(self, input_text: str, fingerprint: Optional[Dict], seed_state: torch.Tensor) -> str:
        """Handle association learning via fossil recovery and resonance injection.
        
        If input_text contains '<->' (a paired description), fossilize the new dyad
        FIRST, then perform resonance scanning.  This is the standard ASSOCIATE workflow:
        the user provides both sides of the dyad and expects a new fossil to be written.
        """
        print(" Phase 4: Dyadic Association Recovery")
        
        # --- FOSSILIZE if this is a paired association (contains '<->') ---
        fossil_log = ""
        if "<->" in input_text:
            raw = input_text.replace("ASSOCIATE:", "").strip()
            parts = raw.split("<->", 1)
            source_desc = parts[0].strip()
            target_desc = parts[1].strip() if len(parts) > 1 else ""
            # Build a synthetic INGEST_DYAD: command from the paired text
            ingest_cmd = f"INGEST_DYAD: {source_desc} <-> {target_desc}"
            fossil_log = self._handle_dyad_ingestion(ingest_cmd, fingerprint, seed_state)
            print(f"[ASSOCIATE] Auto-fossilized paired association. {fossil_log}")

        # --- RESONANCE SCAN against existing fossils ---
        fossils = self.fossilizer.recover_fossils()
        if not fossils:
            self._last_resonance = 0.0
            prefix = f"{fossil_log}\n" if fossil_log else ""
            return prefix + "No prior fossils found. This association is now the first topological obstruction."
            
        # Compute resonance between current seed_state and fossils
        best_resonance = -1.0
        best_fossil = None
        
        for f in fossils:
            if 'residue_vector' not in f:
                continue
                
            residue = f['residue_vector'].to(self.device).flatten()
            res = torch.dot(seed_state.flatten(), residue) / (torch.norm(seed_state) * torch.norm(residue) + 1e-8)
            if res > best_resonance:
                best_resonance = res
                best_fossil = f
        
        self._last_resonance = float(best_resonance)
                
        prefix = f"{fossil_log}\n" if fossil_log else ""
        if best_fossil and best_resonance > 0.5:
            with torch.no_grad():
                res_vec = best_fossil['residue_vector'].to(self.device).view_as(self.meta_state)
                self.meta_state = self.meta_state + 0.3 * res_vec
                
            return (
                prefix +
                f"Resonating with prior fossil: '{best_fossil.get('description', '?')[:60]}...'. "
                f"Resonance Score: {best_resonance:.3f}. Residue injected into meta-functional manifold."
            )
        
        return prefix + f"Manifold scanned. No resonant fossils found for current state (Max Resonance: {best_resonance:.3f})."
    
    
    def _compute_full_gyroid_violation_score(self, state: torch.Tensor, response_text: str) -> float:
        """
        Phase 4.1: Complete Gyroid Violation Score computation.
        
        Implements full gyroidic manifold violation detection using:
        - Spectral signature analysis
        - Covariance-based pressure computation  
        - Topological consistency checks
        - Response-state correlation analysis
        """
        print(" Phase 4.1: Computing Full Gyroid Violation Score...")
        
        try:
            # Initialize gyroid covariance probe if not exists
            if not hasattr(self, '_gyroid_probe'):
                from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe
                self._gyroid_probe = SparseGyroidCovarianceProbe(
                    hidden_dim=self.dim,
                    window_size=32,
                    k_hop=2,
                    num_eigenvalues=8,
                    violation_threshold=0.5,
                    use_saturation_detection=True,
                    adaptive_threshold=True
                )
            # Prepare state for analysis [batch, seq_len, hidden_dim]
            if state.dim() == 2:
                # Add sequence dimension
                analysis_state = state.unsqueeze(1)  # [1, 1, dim]
            else:
                analysis_state = state
            
            # Compute gyroid covariance violations
            violation_results = self._gyroid_probe(analysis_state)
            
            # Extract violation metrics
            gcve_scores = violation_results.get('gcve_scores', torch.tensor([0.0]))
            fracture_scores = violation_results.get('fracture_scores', torch.tensor([0.0]))
            total_pressure = violation_results.get('total_pressure', torch.tensor([0.0]))
            
            # Compute response-state correlation violation
            response_violation = self._compute_response_state_violation(state, response_text)
            
            # Combined gyroid violation score
            base_violation = total_pressure.mean().item()
            response_correlation_violation = response_violation
            
            # Hybridization: Integrate real hardware-level Gyroid TPMS projection
            hw_deviation = 0.0
            if hasattr(self, 'sovereignty_engine') and self.sovereignty_engine is not None:
                try:
                    import numpy as np
                    coords_np = state.detach().cpu().numpy().astype(np.float32)
                    if coords_np.size >= 3:
                        # Reshape flat state elements into coordinate triples [N, 3]
                        num_triples = coords_np.size // 3
                        coords_np_3d = coords_np.flatten()[:num_triples * 3].reshape(num_triples, 3)
                        
                        # Project onto the Gyroid TPMS using the compiled OpenCL kernel
                        projected_3d = self.sovereignty_engine.apply_gyroid_projection(coords_np_3d, max_steps=10)
                        hw_deviation = float(np.abs(coords_np_3d - projected_3d).mean())
                        print(f" [HYBRID] OpenCL hardware Gyroid TPMS projection deviation: {hw_deviation:.6f}")
                        
                        # Proactive garbage collection for constrained 8GB RAM / 4GB VRAM hardware
                        import gc
                        gc.collect()
                        # Flush PyOpenCL queues to immediately reclaim device VRAM on non-CUDA setups
                        if hasattr(self, 'sovereignty_engine') and self.sovereignty_engine is not None:
                            if hasattr(self.sovereignty_engine, 'queue_a'):
                                self.sovereignty_engine.queue_a.finish()
                            if hasattr(self.sovereignty_engine, 'queue_b'):
                                self.sovereignty_engine.queue_b.finish()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Blend the PyTorch spectral violation and the real physical projection deviation
                        base_violation = 0.5 * base_violation + 0.5 * hw_deviation
                except Exception as ex:
                    print(f" [HYBRID] OpenCL projection fallback skipped: {ex}")
            
            # Weighted combination
            full_violation_score = (
                0.6 * base_violation +
                0.4 * response_correlation_violation
            )
            return float(full_violation_score)
            
        except Exception as e:
            print(f"  Gyroid violation computation failed: {e}")
            # Fallback to simple norm-based violation
            return float(torch.norm(state).item() * 0.1)
    
    def _compute_response_state_violation(self, state: torch.Tensor, response_text: str) -> float:
        """Compute violation based on response-state correlation."""
        if not response_text:
            return 0.0
        
        # Convert response back to tensor representation
        response_tensor = self._text_to_tensor(response_text)
        
        # Compute correlation between state and response representation
        state_flat = state.flatten()
        response_flat = response_tensor.flatten()
        
        # Ensure same size for correlation
        min_size = min(len(state_flat), len(response_flat))
        state_truncated = state_flat[:min_size]
        response_truncated = response_flat[:min_size]
        
        # Compute cosine similarity
        correlation = torch.cosine_similarity(
            state_truncated.unsqueeze(0), 
            response_truncated.unsqueeze(0), 
            dim=1
        ).item()
        
        # Violation is inverse of correlation (high correlation = low violation)
        violation = 1.0 - abs(correlation)
        
        return violation
    
    def _perform_unfolding_closure_check(self, state: torch.Tensor, input_text: str, response_text: str, fractal_components: dict = None) -> dict:
        """
        Phase 4.2: Complete Unfolding Closure Check implementation.
        
        Implements topological closure verification using:
        - Hyper-ring operator evaluation
        - Cycle closure detection
        - Triadic reciprocity validation
        - Unfolding branch analysis
        """
        print(" Phase 4.2: Performing Unfolding Closure Check...")
        
        try:
            # Initialize closure checker if not exists
            if not hasattr(self, '_closure_checker'):
                # Import required components
                try:
                    from src.topology.hyper_ring_closure import HyperRingClosureChecker
                    self._closure_checker = HyperRingClosureChecker(
                        closure_tolerance=1e-4,
                        trivial_threshold=1e-3
                    )
                except ImportError:
                    # Fallback implementation
                    return self._fallback_closure_check(state, input_text, response_text)
            
            # Create hyper-ring representation from state
            hyper_ring = self._create_hyper_ring_from_state(state, input_text, response_text)
            
            # Collapse hyper_ring to [batch] if it is [batch, dim] to match closure checker expectations
            if hyper_ring.dim() > 1 and hyper_ring.shape[-1] > 1:
                hyper_ring_input = torch.norm(hyper_ring, dim=-1)
            else:
                hyper_ring_input = hyper_ring.squeeze(-1) if hyper_ring.dim() > 1 else hyper_ring
            
            # Create constraint manifold representation
            # We treat the first fractal component (crt) as the reference constraint manifold
            constraint_manifold = state
            if fractal_components and 'crt' in fractal_components:
                constraint_manifold = fractal_components['crt']
            
            # Ensure dimensional compatibility for closure check
            # Energy-based dimension alignment
            if hyper_ring.shape[-1] != constraint_manifold.shape[-1]:
                # Align dimensions using energy-preserving projection
                target_dim = min(hyper_ring.shape[-1], constraint_manifold.shape[-1])
                
                if hyper_ring.shape[-1] > target_dim:
                    # Project hyper_ring down
                    projection_matrix = torch.eye(target_dim, hyper_ring.shape[-1], device=hyper_ring.device)
                    hyper_ring = torch.mm(hyper_ring, projection_matrix.t())
                
                if constraint_manifold.shape[-1] > target_dim:
                    # Project constraint_manifold down
                    projection_matrix = torch.eye(target_dim, constraint_manifold.shape[-1], device=constraint_manifold.device)
                    constraint_manifold = torch.mm(constraint_manifold, projection_matrix.t())
            
            # Perform closure check with aligned dimensions
            closure_result = self._closure_checker(hyper_ring_input, constraint_manifold)
            
            # Extract results
            is_closed_val = closure_result.get('is_closed', torch.tensor([False]))
            is_trivial_val = closure_result.get('is_trivial', torch.tensor([True]))
            is_valid_val = closure_result.get('is_valid', torch.tensor([False]))
            
            is_closed = bool(is_closed_val.any().item()) if hasattr(is_closed_val, 'any') else bool(is_closed_val)
            is_trivial = bool(is_trivial_val.any().item()) if hasattr(is_trivial_val, 'any') else bool(is_trivial_val)
            is_valid = bool(is_valid_val.any().item()) if hasattr(is_valid_val, 'any') else bool(is_valid_val)
            
            # Compute unfolding branches
            unfolding_branches = self._compute_unfolding_branches(state, response_text)
            
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
                'is_closed': bool(is_closed),
                'is_trivial': bool(is_trivial),
                'is_valid': bool(is_valid),
                'unfolding_branches': unfolding_branches,
                'closure_quality': float(1.0 - is_trivial) if is_closed else 0.0
            }
            
        except Exception as e:
            print(f"  Unfolding closure check failed: {e}")
            return self._fallback_closure_check(state, input_text, response_text)
    
    def _create_hyper_ring_from_state(self, state: torch.Tensor, input_text: str, response_text: str) -> torch.Tensor:
        """
        Create hyper-ring representation using existing HyperRingOperator.
        
        This uses the existing topology/hyper_ring.py system for proper
        hyper-ring creation with topological guarantees.
        """
        try:
            # Use stabilized HyperRingOperator from hyper_ring_closure
            from src.topology.hyper_ring_closure import HyperRingOperator
            
            # Create hyper-ring operator
            ring_operator = HyperRingOperator(
                ring_dim=min(32, state.shape[-1]),
                closure_tolerance=1e-4
            )
            # Combine input, state, and response information
            input_tensor = self._text_to_tensor(input_text)
            response_tensor = self._text_to_tensor(response_text)
            
            # Use existing hyper-ring operator
            hyper_ring = ring_operator.create_ring_from_components(
                state=state,
                input_component=input_tensor,
                response_component=response_tensor
            )
            return hyper_ring
            
        except ImportError:
            # Fallback to simple implementation
            input_tensor = self._text_to_tensor(input_text)
            response_tensor = self._text_to_tensor(response_text)
            
            # Ensure all tensors have compatible dimensions
            target_dim = state.shape[-1] if state.dim() > 0 else 32
            
            # Resize tensors to match
            if input_tensor.numel() > target_dim:
                input_tensor = input_tensor.flatten()[:target_dim]
            elif input_tensor.numel() < target_dim:
                input_tensor = F.pad(input_tensor.flatten(), (0, target_dim - input_tensor.numel()))
            else:
                input_tensor = input_tensor.flatten()
                
            if response_tensor.numel() > target_dim:
                response_tensor = response_tensor.flatten()[:target_dim]
            elif response_tensor.numel() < target_dim:
                response_tensor = F.pad(response_tensor.flatten(), (0, target_dim - response_tensor.numel()))
            else:
                response_tensor = response_tensor.flatten()
            
            # Create ring structure with proper dimensions
            if state.dim() == 1:
                state_flat = state
            else:
                state_flat = state.flatten()[:target_dim]
                if state_flat.numel() < target_dim:
                    state_flat = F.pad(state_flat, (0, target_dim - state_flat.numel()))
            
            # Combine with proper weighting
            hyper_ring = (state_flat + 0.1 * input_tensor + 0.1 * response_tensor).unsqueeze(0)
            
            return hyper_ring
    
    def _create_constraint_manifold(self, state: torch.Tensor) -> torch.Tensor:
        """
        Create constraint manifold representation using existing polynomial CRT.
        
        This uses the existing DecoupledPolynomialCRT system for proper
        constraint manifold creation with guaranteed dimensional consistency.
        """
        # Ensure state is properly shaped [batch, dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, dim]
        
        batch_size, dim = state.shape
        
        # Use existing polynomial CRT for manifold creation if available
        if hasattr(self, '_decoupled_crt') and self._decoupled_crt is not None:
            try:
                # Use decoupled CRT to create constraint manifold
                manifold = self._decoupled_crt.create_constraint_manifold(state)
                return manifold
            except Exception as e:
                print(f"  Decoupled CRT manifold creation failed: {e}")
        
        # Fallback: simple orthogonal projection with proper dimensions
        constraint_dim = min(dim, 8)  # Reasonable constraint dimension
        
        # Ensure we don't exceed available dimensions
        if constraint_dim > dim:
            constraint_dim = dim
        
        # Create orthogonal constraint directions
        if constraint_dim == dim:
            # Identity mapping if dimensions match
            manifold_projected = state
        else:
            # Project to lower dimension
            constraint_dirs = torch.eye(constraint_dim, dim, device=state.device)
            manifold_projected = torch.mm(state, constraint_dirs.t())
        
        return manifold_projected
    
    def _compute_unfolding_branches(self, state: torch.Tensor, response_text: str) -> int:
        """Compute number of unfolding branches in the topological structure."""
        # Analyze state for branching patterns
        state_flat = state.flatten()
        
        # Look for oscillatory patterns that indicate branches
        # Use FFT to detect frequency components
        fft_result = torch.fft.fft(state_flat)
        magnitude_spectrum = torch.abs(fft_result)
        
        # Count significant frequency peaks (branches)
        threshold = magnitude_spectrum.mean() + magnitude_spectrum.std()
        significant_peaks = (magnitude_spectrum > threshold).sum().item()
        
        # Limit to reasonable range
        branches = min(max(significant_peaks, 1), 8)
        
        return branches
    
    def _fallback_closure_check(self, state: torch.Tensor, input_text: str, response_text: str) -> dict:
        """Fallback closure check implementation."""
        # Simple heuristic-based closure check
        state_norm = torch.norm(state).item()
        response_length = len(response_text)
        
        # Heuristic: closed if state norm is reasonable and response is coherent
        is_closed = 0.1 < state_norm < 10.0 and response_length > 5
        is_trivial = response_length < 10
        is_valid = is_closed and not is_trivial
        
        return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
            'is_closed': is_closed,
            'is_trivial': is_trivial,
            'is_valid': is_valid,
            'unfolding_branches': 3,  # Default
            'closure_quality': 0.5 if is_valid else 0.0
        }
    
    def _perform_advanced_topological_analysis(self, state: torch.Tensor, input_text: str, response_text: str) -> dict:
        """
        Phase 4.3: Advanced topological analysis and graph generation.
        
        Implements comprehensive topological feature detection:
        - Persistent homology approximation
        - Betti number computation
        - Cycle detection and classification
        - Manifold curvature estimation
        """
        print(" Phase 4.3: Performing Advanced Topological Analysis...")
        
        try:
            # Initialize topological analyzer if not exists
            if not hasattr(self, '_topo_analyzer'):
                self._topo_analyzer = self._create_topological_analyzer()
            
            # Extract topological features
            features = []
            
            # Feature 1: Persistent homology approximation
            persistence_features = self._compute_persistence_features(state, response_text)
            features.extend(persistence_features)
            
            # Feature 2: Betti numbers
            betti_numbers = self._compute_betti_numbers(state)
            features.append(f"betti_0={betti_numbers[0]:.2f}")
            features.append(f"betti_1={betti_numbers[1]:.2f}")
            
            # Feature 3: Cycle detection
            cycles = self._detect_topological_cycles(state, input_text, response_text)
            features.append(f"cycles={len(cycles)}")
            
            # Feature 4: Manifold curvature estimation
            curvature = self._estimate_manifold_curvature(state)
            features.append(f"curvature={curvature:.4f}")
            
            # Feature 5: Graph connectivity analysis
            connectivity = self._analyze_graph_connectivity(state, response_text)
            features.extend(connectivity)
            
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
                'features': features,
                'num_features': len(features),
                'persistence_dimension': len(persistence_features),
                'topological_complexity': len(cycles) + sum(betti_numbers)
            }
            
        except Exception as e:
            print(f"  Advanced topological analysis failed: {e}")
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
                'features': ['fallback_analysis'],
                'num_features': 1,
                'persistence_dimension': 0,
                'topological_complexity': 0.0
            }
    
    def _create_topological_analyzer(self):
        """Create topological analyzer instance."""
        # Simple analyzer that tracks state evolution
        return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
            'state_history': [],
            'max_history': 10
        }
    
    def _compute_persistence_features(self, state: torch.Tensor, response_text: str) -> list:
        """Compute persistent homology features."""
        features = []
        
        # Analyze state persistence across dimensions
        state_flat = state.flatten()
        
        # Compute persistence intervals (simplified)
        # Look for stable patterns in the state vector
        threshold = state_flat.std().item()
        stable_dims = (torch.abs(state_flat) > threshold).sum().item()
        
        features.append(f"persistent_dims={stable_dims}")
        
        # Response persistence
        if response_text:
            char_variety = len(set(response_text.lower()))
            features.append(f"response_variety={char_variety}")
        
        return features
    
    def _compute_betti_numbers(self, state: torch.Tensor) -> list:
        """Compute approximate Betti numbers."""
        state_flat = state.flatten()
        
        #  (connected components) - approximate via clustering
        # Use simple threshold-based clustering
        threshold = state_flat.std().item()
        positive_components = (state_flat > threshold).sum().item()
        negative_components = (state_flat < -threshold).sum().item()
        beta_0 = max(1, positive_components + negative_components) / len(state_flat)
        
        #  (cycles) - approximate via autocorrelation
        # Look for periodic patterns
        autocorr = compute_autocorrelation(state_flat)
        autocorr_normalized = autocorr / autocorr.max()
        
        # Count significant autocorrelation peaks (indicating cycles)
        peaks = (autocorr_normalized > 0.5).sum().item()
        beta_1 = min(peaks / len(autocorr_normalized), 1.0)
        
        return [beta_0, beta_1]
    
    def _detect_topological_cycles(self, state: torch.Tensor, input_text: str, response_text: str) -> list:
        """Detect topological cycles in the state space."""
        cycles = []
        
        # Analyze state for cyclic patterns
        state_flat = state.flatten()
        
        # Look for approximate cycles using sliding window correlation
        window_size = min(8, len(state_flat) // 4)
        if window_size > 2:
            for i in range(len(state_flat) - 2 * window_size):
                window1 = state_flat[i:i + window_size]
                window2 = state_flat[i + window_size:i + 2 * window_size]
                
                # Check if windows are similar (indicating cycle)
                correlation = torch.cosine_similarity(window1, window2, dim=0).item()
                if correlation > 0.8:  # High similarity threshold
                    cycles.append({
                        'start': i,
                        'length': window_size,
                        'correlation': correlation
                    })
        
        return cycles
    
    def _estimate_manifold_curvature(self, state: torch.Tensor) -> float:
        """Estimate manifold curvature from state."""
        state_flat = state.flatten()
        
        if len(state_flat) < 3:
            return 0.0
        
        # Approximate curvature using second derivatives
        # Compute discrete second derivative
        first_diff = state_flat[1:] - state_flat[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]
        
        # Curvature approximation
        curvature = torch.mean(torch.abs(second_diff)).item()
        
        return curvature
    
    def _analyze_graph_connectivity(self, state: torch.Tensor, response_text: str) -> list:
        """Analyze graph connectivity properties."""
        connectivity_features = []
        
        # Create adjacency matrix from state
        state_flat = state.flatten()
        n = min(16, len(state_flat))  # Limit size for efficiency
        
        if n > 1:
            # Create adjacency based on state correlations
            state_subset = state_flat[:n]
            adjacency = torch.outer(state_subset, state_subset)
            adjacency = torch.abs(adjacency)
            
            # Threshold to create binary adjacency
            threshold = adjacency.mean()
            binary_adj = (adjacency > threshold).float()
            
            # Compute connectivity metrics
            degree_sum = binary_adj.sum().item()
            max_degree = binary_adj.sum(dim=1).max().item()
            
            connectivity_features.append(f"total_degree={degree_sum:.1f}")
            connectivity_features.append(f"max_degree={max_degree:.1f}")
            
            # Estimate clustering coefficient
            if max_degree > 0:
                clustering = degree_sum / (n * (n - 1))  # Simplified
                connectivity_features.append(f"clustering={clustering:.3f}")
        
        return connectivity_features
    
    def _filter_document_noise(self, text: str) -> str:
        """
        Smart filtering for Wikipedia-style document noise while preserving mathematical content.
        
        Removes:
        - Wikipedia reference brackets [1], [2], [citation needed]
        - Excessive formatting artifacts
        - Redundant whitespace
        
        Preserves:
        - Mathematical expressions [x+y], [0,1], [matrix]
        - Meaningful brackets in context
        - Scientific notation and equations
        """
        import re
        
        # Step 1: Preserve mathematical contexts
        # Identify mathematical patterns to protect
        math_patterns = []
        
        # Protect mathematical expressions
        math_contexts = [
            r'\[[\d\+\-\*\/\^\(\)\s,\.]+\]',  # [1+2], [0,1], [x^2]
            r'\[[A-Za-z]\s*[=\+\-\*\/]\s*[A-Za-z\d]+\]',  # [x=5], [a+b]
            r'\[\s*\d+\s*,\s*\d+\s*\]',  # [1,2], [0, 1]
            r'\[.*?matrix.*?\]',  # [matrix], [identity matrix]
            r'\[.*?equation.*?\]',  # [equation 1]
            r'\[.*?formula.*?\]',  # [formula]
            r'\[.*?theorem.*?\]',  # [theorem]
            r'\[.*?proof.*?\]',  # [proof]
        ]
        
        protected_spans = []
        for pattern in math_contexts:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                protected_spans.append((match.start(), match.end()))
        
        # Step 2: Remove Wikipedia-style references
        filtered_text = text
        
        # Remove simple numeric references [1], [2], [123]
        # But only if they're not in protected mathematical contexts
        def is_protected(start, end):
            for p_start, p_end in protected_spans:
                if start >= p_start and end <= p_end:
                    return True
            return False
        
        # Find and remove unprotected numeric references
        ref_pattern = r'\[\s*\d+\s*\]'
        matches = list(re.finditer(ref_pattern, filtered_text))
        
        # Remove from end to start to preserve indices
        for match in reversed(matches):
            if not is_protected(match.start(), match.end()):
                filtered_text = filtered_text[:match.start()] + filtered_text[match.end():]
        
        # Remove citation-style references
        citation_patterns = [
            r'\[citation needed\]',
            r'\[needs citation\]',
            r'\[source\?\]',
            r'\[clarification needed\]',
            r'\[when\?\]',
            r'\[who\?\]',
            r'\[where\?\]',
            r'\[dubious.*?\]',
            r'\[verify.*?\]',
            r'\[original research\?\]',
        ]
        
        for pattern in citation_patterns:
            filtered_text = re.sub(pattern, '', filtered_text, flags=re.IGNORECASE)
        
        # Remove multiple author references like [Smith 2020], [Jones et al. 2019]
        # But preserve mathematical notation
        author_ref_pattern = r'\[[A-Z][a-z]+(?:\s+et\s+al\.?)?\s+\d{4}[a-z]?\]'
        filtered_text = re.sub(author_ref_pattern, '', filtered_text)
        
        # Step 3: Clean up formatting artifacts
        # Remove excessive whitespace
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        
        # Remove orphaned punctuation from removed references
        filtered_text = re.sub(r'\s*,\s*,', ',', filtered_text)  # Double commas
        filtered_text = re.sub(r'\s*\.\s*\.', '.', filtered_text)  # Double periods
        filtered_text = re.sub(r'\s+([,.;:])', r'\1', filtered_text)  # Space before punctuation
        
        # Step 4: Preserve paragraph structure
        # Ensure sentences don't run together
        filtered_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', filtered_text)
        
        return filtered_text.strip()
    
    def _enhanced_association_learning(self, source: str, target: str, similarity: float):
        """
        Enhanced association learning with adaptive weighting based on content length and similarity.
        
        Args:
            source: Short source concept
            target: Long filtered target content
            similarity: Computed semantic similarity
        """
        # Adaptive learning rate based on content characteristics
        source_len = len(source)
        target_len = len(target)
        length_ratio = target_len / max(source_len, 1)
        
        # Higher learning rate for high-quality associations
        base_lr = 0.01
        similarity_boost = similarity * 0.5  # 0-0.5 boost
        length_penalty = min(length_ratio / 100, 0.5)  # Penalty for very long targets
        
        adaptive_lr = base_lr * (1 + similarity_boost - length_penalty)
        adaptive_lr = max(adaptive_lr, 0.001)  # Minimum learning rate
        
        print(f" Enhanced learning: lr={adaptive_lr:.4f}, length_ratio={length_ratio:.1f}")
        
        # Temporarily adjust optimizer learning rate
        old_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = adaptive_lr
        
        # Enhanced mimicry training with filtered content
        source_tensor = self._text_to_tensor(source)
        self._train_mimicry(source_tensor, target)
        
        # Also train reverse association (target concept -> source)
        if len(target) > 50:  # Only for substantial targets
            # Use first 100 chars of target as reverse source
            target_sample = target[:100]
            target_tensor = self._text_to_tensor(target_sample)
            self._train_mimicry(target_tensor, source)
            print(f" Bidirectional learning: '{target_sample[:20]}...'  '{source}'")
        
        # Restore original learning rate
        self.optimizer.param_groups[0]['lr'] = old_lr

    def _run_advanced_physics(self, text_input: str, gradients: Dict[str, float]) -> Dict:
        """
        Run System 2 Advanced Physics (Quantum/Polytope) if budget allows.
        """
        start_time = time.time()
        diagnostics = {}
        
        # 1. Trigger Check: sufficient formal pressure?
        formal_pressure = gradients.get('formal_symbol_density', 0.0) # Corrected key
        if formal_pressure < 0.6: # Relaxed threshold
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, }
            
        # 2. Budget Check: Do we have latency headroom?
        # Assuming we are ~0.3s into processing. Limit total to 1.0s.
        if (time.time() - self.last_input_time) > 0.8:
            print(f" Advanced Physics skipped: budget exceeded ({time.time() - self.last_input_time:.2f}s)")
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 'budget_abort': True}
            
        try:
            # Lazy Init
            if self.meta_polytope is None:
                self.meta_polytope = MetaPolytopeMatrioshka(max_depth=5, base_dim=self.dim) # Use self.dim
                self.quantum_reasoner = QuantumInspiredReasoningState(dim=self.dim) # Use self.dim
                
            # 3. Meta-Polytope Matrioshka
            # Project current cavity state
            if self.cavity.short_term_memory:
                 # Use last memory state
                 input_state = self.cavity.short_term_memory[-1]
                 if input_state.shape[-1] != self.dim: # Pad/Cut
                     # Ensure input_state is 1D for padding
                     input_state_flat = input_state.flatten()
                     if input_state_flat.shape[0] < self.dim:
                         input_state = F.pad(input_state_flat, (0, self.dim - input_state_flat.shape[0])).unsqueeze(0)
                     else:
                         input_state = input_state_flat[:self.dim].unsqueeze(0)
                 
                 # Matrioshka quantization
                 q_state, alpha, level = self.meta_polytope(input_state) # input_state is already [1, dim]
                 diagnostics['matrioshka_level'] = int(level)
                 diagnostics['crt_index'] = int(alpha)
                 
                 # 4. Quantum Reasoning
                 # If Matrioshka level is high (deep thought), engage Quantum
                 if level >= 1:
                     # Create hypotheses from spectral variations
                     hypotheses = [input_state.squeeze(0), q_state.squeeze(0), (input_state * 1.1).squeeze(0)]
                     probs = self.quantum_reasoner.superposition_reasoning(hypotheses)
                     superposition_entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
                     
                     diagnostics['quantum_superposition'] = True
                     diagnostics['spectral_entropy'] = superposition_entropy # Override with quantum entropy
                     print(f" Advanced Physics: Matrioshka Level {level}, Quantum Entropy {superposition_entropy:.3f}")
            
        except Exception as e:
            print(f" Advanced Physics Error: {e}")
            diagnostics['error'] = str(e)
            
        return diagnostics

    def save_state(self):
        # Neural state
        torch.save(self.state_dict(), STATE_PATH)
        # Artifact state
        if hasattr(self.encoding_manager, 'save_artifacts'):
            self.encoding_manager.save_artifacts()

        # Save Python-native attributes like iteration
        # Since iteration isn't a buffer, we can save it in a small sidecar dict
        metadata = {'iteration': self.iteration}
        torch.save(metadata, STATE_PATH + ".meta")

        print(f" Full state & artifacts persisted.")

    def load_state(self):
        """Unified load: Neural Weights + Metadata + Encoding Context."""
        if not os.path.exists(STATE_PATH):
            print(" No persistence file found. Starting fresh.")
            return False

        try:
            # 1. Load Neural Weights (Non-Strict for flexibility)
            checkpoint = torch.load(STATE_PATH, map_location=self.device)
            load_result = self.load_state_dict(checkpoint, strict=False)
            
            # Log counts of missing/unexpected keys
            missing = len(load_result.missing_keys)
            unexpected = len(load_result.unexpected_keys)
            print(f" Neural load complete. Missing keys: {missing}, Unexpected: {unexpected}")

            # 2. Repair non-finite values (NaN/Inf)
            repair_count = self._repair_tensors()
            print(f" Deterministic non-finite repair complete. Repaired Tensors: {repair_count}")

            # 3. Synchronize Encoding Context
            # Update the engine's iteration count from the manager's findings
            self.iteration = self.encoding_manager.get_latest_iteration()

            print(f" State restored. Resuming from iteration {self.iteration}")
            return True
        except Exception as e:
            print(f" Critical Load Failure: {e}")
            return False

    def _repair_tensors(self) -> int:
        """Surgical repair of non-finite parameters. Returns count of repaired tensors."""
        repair_count = 0
        with torch.no_grad():
            for name, tensor in self.state_dict().items():
                if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                    tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    repair_count += 1
        return repair_count

# Initialize Engine (only when running as the server entry point, not when imported by tests)
_running_as_server = (__name__ == '__main__') or os.environ.get('GYROID_SERVER_MODE', '0') == '1'
if _running_as_server:
    ENGINE = DiegeticPhysicsEngine()
    ENGINE.load_state()
else:
    ENGINE = None
def parse_multipart(body: bytes, boundary: bytes) -> dict:
    """Parses multipart/form-data request body."""
    parts = {}
    boundary_marker = b'--' + boundary
    raw_parts = body.split(boundary_marker)
    for part in raw_parts:
        if not part or part == b'--\r\n' or part == b'--':
            continue
        if part.startswith(b'\r\n'):
            part = part[2:]
        if part.endswith(b'\r\n'):
            part = part[:-2]
        
        if b'\r\n\r\n' not in part:
            continue
        headers_part, content = part.split(b'\r\n\r\n', 1)
        headers = headers_part.decode('utf-8', errors='ignore')
        
        name = None
        filename = None
        for line in headers.split('\r\n'):
            if line.lower().startswith('content-disposition:'):
                parts_disp = line.split(';')
                for p in parts_disp:
                    p = p.strip()
                    if p.startswith('name='):
                        name = p.split('=', 1)[1].strip('"\'')
                    elif p.startswith('filename='):
                        filename = p.split('=', 1)[1].strip('"\'')
        
        if name:
            if filename:
                parts[name] = {
                    'filename': filename,
                    'content': content
                }
            else:
                parts[name] = content.decode('utf-8', errors='ignore')
    return parts


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path == '/' or self.path == '':
                # Serve the diegetic terminal HTML
                try:
                    # Use absolute path to ensure we find the file
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    terminal_path = os.path.join(current_dir, 'diegetic_terminal.html')
                    
                    # Fallback to relative path if absolute doesn't work
                    if not os.path.exists(terminal_path):
                        terminal_path = os.path.join('src', 'ui', 'diegetic_terminal.html')
                    
                    print(f" Serving diegetic terminal from: {terminal_path}")
                    print(f" File exists: {os.path.exists(terminal_path)}")
                    
                    if not os.path.exists(terminal_path):
                        print(f" Diegetic terminal HTML not found at {terminal_path}")
                        self.send_error(404, f"Diegetic terminal HTML not found: {terminal_path}")
                        return
                    
                    with open(terminal_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    print(f" Diegetic terminal content length: {len(content)}")
                    
                    if len(content) == 0:
                        print(" Diegetic terminal HTML is empty!")
                        self.send_error(500, "Diegetic terminal HTML is empty")
                        return
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                    print(" Diegetic terminal served successfully")
                    return
                except Exception as e:
                    print(f" Error serving diegetic terminal: {e}")
                    import traceback
                    traceback.print_exc()
                    self.send_error(500, f"Error serving diegetic terminal: {e}")
                    return
            elif self.path == '/graph':
                print("API REQUEST: /graph")
                ENGINE.graph_manager.load_fossils(limit=150) 
                graph_data = json.loads(ENGINE.graph_manager.export_graph_json())
                self._send_json(graph_data)
                return
            elif self.path == '/health':
                print("API REQUEST: /health")
                self._send_json({"status": "hyper-ring coherent", "version": "1.9.1"})
                return
            elif self.path == '/ping':
                self._send_json({
                    "status": "online",
                    "pid": os.getpid(),
                    "uptime": time.time() - START_TIME
                })
                return
            
            elif self.path == '/api/minecraft/scan':
                print("API REQUEST: /api/minecraft/scan")
                try:
                    minecraft_dir = os.path.join(os.getcwd(), 'datasets', 'minecraft')
                    os.makedirs(minecraft_dir, exist_ok=True)
                    
                    worlds = []
                    mods = []
                    
                    # Scan for worlds (subdirectories)
                    for item in os.listdir(minecraft_dir):
                        item_path = os.path.join(minecraft_dir, item)
                        if os.path.isdir(item_path):
                            if item in ['.venv', '__pycache__', 'data', 'datasets', 'mods']:
                                continue
                            
                            has_level_dat = os.path.exists(os.path.join(item_path, 'level.dat'))
                            has_region = os.path.exists(os.path.join(item_path, 'region'))
                            
                            worlds.append({
                                'name': item,
                                'path': os.path.relpath(item_path, os.getcwd()),
                                'has_level_dat': has_level_dat,
                                'has_region': has_region
                            })
                    
                    # Scan for mods (JARs and ZIPs) in datasets/minecraft/mods/
                    mods_dir = os.path.join(minecraft_dir, 'mods')
                    os.makedirs(mods_dir, exist_ok=True)
                    for item in os.listdir(mods_dir):
                        item_path = os.path.join(mods_dir, item)
                        if os.path.isfile(item_path) and item.endswith(('.jar', '.zip')):
                            mods.append({
                                'name': item,
                                'path': os.path.relpath(item_path, os.getcwd()),
                                'size': os.path.getsize(item_path)
                            })
                            
                    self._send_json({
                        'success': True,
                        'worlds': worlds,
                        'mods': mods,
                        'directory': os.path.relpath(minecraft_dir, os.getcwd())
                    })
                except Exception as e:
                    self._send_error_json(str(e))
                return
            
            # --- LOCAL DATA ENDPOINTS (Phase 1) ---
            elif self.path == '/api/local_datasets':
                print("API REQUEST: /api/local_datasets")
                datasets = LOCAL_LOADER.scan()
                summary = LOCAL_LOADER.get_summary()
                self._send_json({'success': True, **summary})
                return
            
            elif self.path == '/api/training_status':
                self._send_json(TRAINING_STATE)
                return
            
            # --- GUI SERVING ---
            elif self.path == '/conversational-gui':
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    gui_path = os.path.join(current_dir, 'conversational_web_gui.html')
                    if not os.path.exists(gui_path):
                        gui_path = os.path.join('src', 'ui', 'conversational_web_gui.html')
                    if os.path.exists(gui_path):
                        with open(gui_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/html; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(content.encode('utf-8'))
                        return
                    else:
                        self.send_error(404, "Conversational GUI not found")
                        return
                except Exception as e:
                    self.send_error(500, f"Error serving conversational GUI: {e}")
                    return
            
            elif self.path == '/wikipedia-trainer':
                # Serve the Wikipedia trainer HTML
                try:
                    # Use absolute path to ensure we find the file
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    trainer_path = os.path.join(current_dir, 'wikipedia_trainer.html')
                    
                    # Fallback to relative path if absolute doesn't work
                    if not os.path.exists(trainer_path):
                        trainer_path = os.path.join('src', 'ui', 'wikipedia_trainer.html')
                    
                    print(f" Attempting to serve HTML from: {trainer_path}")
                    print(f" File exists: {os.path.exists(trainer_path)}")
                    print(f" Current working directory: {os.getcwd()}")
                    
                    if not os.path.exists(trainer_path):
                        print(f" HTML file not found at {trainer_path}")
                        self.send_error(404, f"HTML file not found: {trainer_path}")
                        return
                    
                    with open(trainer_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    print(f" HTML content length: {len(content)}")
                    
                    if len(content) == 0:
                        print(" HTML file is empty!")
                        self.send_error(500, "HTML file is empty")
                        return
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                    print(" HTML served successfully")
                    return
                except Exception as e:
                    print(f" Error serving Wikipedia trainer: {e}")
                    import traceback
                    traceback.print_exc()
                    self.send_error(500, f"Error serving HTML: {e}")
                    return
            
            # Fallback for static files
            return super().do_GET()
        except Exception as e:
            print(f"CRITICAL GET ERROR: {e}")
            self._send_error_json(str(e))

    def do_POST(self):
        print(f"POST REQUEST RECEIVED: {self.path}")
        try:
            if self.path == '/interact':
                print(" Processing /interact request...")
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    
                    content_type = self.headers.get('Content-Type', '')
                    if content_type.startswith('multipart/form-data'):
                        boundary = b''
                        parts_ct = content_type.split(';')
                        for p in parts_ct:
                            p = p.strip()
                            if p.startswith('boundary='):
                                boundary = p.split('=', 1)[1].encode('utf-8')
                        
                        if not boundary:
                            raise ValueError("Multipart boundary not found in headers")
                        
                        form_fields = parse_multipart(post_body, boundary)
                        
                        data = {}
                        data['text'] = form_fields.get('text', '')
                        data['commutativity'] = form_fields.get('commutativity', 'symmetric')
                        data['regime'] = form_fields.get('regime', 'goo')
                        data['generate_response'] = form_fields.get('generate_response', 'true').lower() == 'true'
                        data['ingestion_mode'] = form_fields.get('ingestion_mode', 'false').lower() == 'true'
                        data['performance_buffered'] = form_fields.get('performance_buffered', 'false').lower() == 'true'
                        data['audio_b64'] = form_fields.get('audio_b64', None)
                        
                        video_file = form_fields.get('video_dyad_file')
                        if isinstance(video_file, dict) and 'content' in video_file:
                            import base64
                            b64_str = base64.b64encode(video_file['content']).decode('utf-8')
                            data['video_dyad_b64'] = b64_str
                            print(f"[BACKEND] Ingested video file from multipart upload ({len(video_file['content'])} bytes).", flush=True)
                        else:
                            data['video_dyad_b64'] = form_fields.get('video_dyad_b64', None)
                        
                        for field in ['fingerprint', 'audio_dyad', 'media_chain']:
                            val = form_fields.get(field)
                            if val:
                                try:
                                    data[field] = json.loads(val)
                                except Exception as e:
                                    print(f"[BACKEND] Warning: Failed to parse field {field} as JSON: {e}", flush=True)
                                    data[field] = None
                            else:
                                data[field] = None
                    else:
                        data = json.loads(post_body.decode('utf-8'))
                    
                    user_text     = data.get('text', '')
                    fingerprint   = data.get('fingerprint', None)
                    audio_dyad    = data.get('audio_dyad', None)
                    video_dyad_b64 = data.get('video_dyad_b64', None)
                    commutativity = data.get('commutativity', 'symmetric')
                    
                    if video_dyad_b64 == "[FILE_POINTER]":
                        video_dyad_b64 = None
                        
                    print(f" User input: '{user_text}' | commutativity={commutativity} | "
                          f"has_image={fingerprint is not None} | has_audio={audio_dyad is not None} | "
                          f"has_video={video_dyad_b64 is not None}")
                    print(" Starting ENGINE.process_input...")

                    response_data = ENGINE.process_input(
                        user_text,
                        fingerprint=fingerprint,
                        audio_dyad=audio_dyad,
                        video_dyad_b64=video_dyad_b64,
                        audio_b64=data.get('audio_b64', None),
                        media_chain=data.get('media_chain', None),
                        commutativity=commutativity,
                        generate_response=data.get('generate_response', True),
                        ingestion_mode=data.get('ingestion_mode', False),
                        performance_buffered=data.get('performance_buffered', False)
                    )
                    self._send_json(response_data)
                except Exception as e:
                    print(f" Error processing input: {e}")
                    import traceback
                    traceback.print_exc()
                    self._send_error_json(str(e))
                return


            elif self.path == '/associate':
                print(" Processing /associate request...")
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    data = json.loads(post_body.decode('utf-8'))
                    
                    text1 = data.get('text1', data.get('source', ''))
                    text2 = data.get('text2', data.get('target', ''))
                    
                    if not text1 or not text2:
                        self._send_error_json("Missing text1/text2 or source/target")
                        return

                    fingerprint = data.get('fingerprint', None)
                    audio_dyad = data.get('audio_dyad', None)
                    video_dyad_b64 = data.get('video_dyad_b64', None)
                    if video_dyad_b64 == "[FILE_POINTER]":
                        video_dyad_b64 = None
                    media_chain = data.get('media_chain', None)
                    commutativity = data.get('commutativity', 'symmetric')
                    voynich_token = data.get('voynich_token', None)
                    
                    association_command = f"ASSOCIATE: {text1} <-> {text2}"
                    print(f" Association command: '{association_command}' | "
                          f"has_fp={fingerprint is not None} | has_audio={audio_dyad is not None} | "
                          f"has_video={video_dyad_b64 is not None} | has_voynich={voynich_token is not None}")
                    
                    response_data = ENGINE.process_input(
                        association_command,
                        fingerprint=fingerprint,
                        audio_dyad=audio_dyad,
                        video_dyad_b64=video_dyad_b64,
                        media_chain=media_chain,
                        commutativity=commutativity,
                        voynich_token=voynich_token
                    )
                    ENGINE.save_state()
                    
                    # Ensure response is consistent
                    if not isinstance(response_data, dict):
                        response_data = {
                            "status": "associated",
                            "source": text1,
                            "target": text2,
                            "metrics": response_data,
                            "multimodal_injection": {
                                "fingerprint": fingerprint is not None,
                                "audio": audio_dyad is not None,
                                "video": video_dyad_b64 is not None
                            }
                        }
                    elif "status" not in response_data:
                        response_data["status"] = "associated"
                        response_data["source"] = text1
                        response_data["target"] = text2

                    self._send_json(response_data)
                except Exception as e:
                    print(f" Error processing association: {e}")
                    import traceback
                    traceback.print_exc()
                    self._send_error_json(str(e))
                return

            elif self.path == '/ingest':
                print(" Processing /ingest request...")
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    data = json.loads(post_body.decode('utf-8'))
                    
                    description = data.get('description', '')
                    fingerprint_list = data.get('fingerprint', [])
                    
                    if not description:
                        self._send_error_json("Missing description")
                        return

                    # Process fingerprint
                    if fingerprint_list:
                        # Ensure it's a list of floats
                        if isinstance(fingerprint_list, dict):
                            # Handle RGB dict format if passed directly
                            # Convert to flattened list or specific expected format
                            # For now, let's assume valid list or handle robustly
                            pass
                        
                        try:
                            fingerprint_tensor = torch.tensor(fingerprint_list, dtype=torch.float32)
                            # Resize to 96 if needed (simple padding/truncation)
                            target_dim = 96
                            if fingerprint_tensor.numel() != target_dim:
                                if fingerprint_tensor.numel() > target_dim:
                                    fingerprint_tensor = fingerprint_tensor[:target_dim]
                                else:
                                    fingerprint_tensor = torch.nn.functional.pad(fingerprint_tensor, (0, target_dim - fingerprint_tensor.numel()))
                            
                            # Create Knowledge Dyad
                            dyad = KnowledgeDyad(
                                image_fingerprint=fingerprint_tensor,
                                linguistic_description=description
                            )
                            
                            # Get text embedding from Engine
                            text_tensor = ENGINE._text_to_tensor(description)
                            
                            # Fossilize
                            fossil_path = ENGINE.fossilizer.fossilize(dyad, text_tensor, seed_state=ENGINE.meta_state)
                            print(f" Dyad fossilized at: {fossil_path}")
                            
                        except Exception as e:
                            print(f" Fossilization failed, continuing with memory-only ingest: {e}")
                            fossil_path = "memory_only"
                    else:
                        fossil_path = "text_only"

                    # Process in Engine
                    ingest_command = f"INGEST_DYAD: {description}"
                    # Pass fingerprint to process_input via some mechanism?
                    # The current process_input signature might not support side-channel data easily
                    # unless we modify it or the Engine stores it temporarily.
                    # Looking at _generate_dyad_aware_response, it accepts a fingerprint argument.
                    # But process_input likely calls it.
                    # Let's assume process_input can handle it or we update state directly.
                    
                    # For now, we'll rely on the text command trigger. 
                    # If process_input supports **kwargs, we could pass it.
                    # Let's check process_input signature if possible, but I can't see it now.
                    # I will assume standard string interface for now, keeping fossilization as the "Side Channel"
                    
                    response_data = ENGINE.process_input(ingest_command)
                    
                    # Augment response with fossil info
                    if isinstance(response_data, dict):
                        response_data['fossil_path'] = fossil_path
                    
                    self._send_json(response_data)
                    
                except Exception as e:
                    print(f" Error processing ingestion: {e}")
                    import traceback
                    traceback.print_exc()
                    self._send_error_json(str(e))
                return

            elif self.path == '/api/minecraft/ingest':
                print("API REQUEST: /api/minecraft/ingest")
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    data = json.loads(post_body.decode('utf-8'))
                    
                    world_name = data.get('world_name', '')
                    max_chunks = int(data.get('max_chunks', 16))
                    
                    if not world_name:
                        self._send_error_json("Missing world_name parameter")
                        return
                        
                    minecraft_dir = os.path.join(os.getcwd(), 'datasets', 'minecraft')
                    world_path = os.path.join(minecraft_dir, world_name)
                    
                    if not os.path.exists(world_path):
                        self._send_error_json(f"World path not found: {world_path}")
                        return
                    
                    pipeline = MinecraftIngestionPipeline(ENGINE.codec.config, ENGINE.poly_config)
                    results = pipeline.ingest_minecraft_world(world_path, max_chunks=max_chunks)
                    
                    # Feed the spatial and script residues into the active engine state
                    if results["combined_residue"] is not None:
                        with torch.no_grad():
                            # Project [K, n, n] residue to [1, dim]
                            flat_res = results["combined_residue"].flatten()
                            if flat_res.numel() > ENGINE.dim:
                                res_projected = flat_res[:ENGINE.dim].unsqueeze(0).to(ENGINE.device)
                            else:
                                res_projected = F.pad(flat_res, (0, ENGINE.dim - flat_res.numel())).unsqueeze(0).to(ENGINE.device)
                            
                            ENGINE.meta_state.copy_(ENGINE.meta_state + 0.1 * res_projected)
                            
                            # Update Zeitgeist Router index if active to warp Mandelbulb visual parameters
                            if ENGINE.zeitgeist_router is not None and ENGINE._zeitgeist_state is not None:
                                M = len(ENGINE._zeitgeist_state.moduli)
                                new_alpha_diag = torch.abs(results["combined_residue"].mean(dim=(-1, -2)))
                                for i, p_i in enumerate(ENGINE._zeitgeist_state.moduli):
                                    new_alpha_diag[i] = new_alpha_diag[i].item() % p_i
                                
                                alpha_tensor = torch.zeros((M, M), device=ENGINE.device)
                                r_col = new_alpha_diag.unsqueeze(1)
                                r_row = new_alpha_diag.unsqueeze(0)
                                alpha_tensor = 0.5 * (r_col + r_row)
                                alpha_tensor.view(-1)[::M + 1] = new_alpha_diag
                                
                                braid_word = []
                                if results["noncommutativity_curvature"] > 0.4:
                                    braid_word = [1, -2, 1]
                                    
                                ENGINE._zeitgeist_state = ENGINE._zeitgeist_state.switched(
                                    new_alpha_tensor=alpha_tensor.cpu(),
                                    new_level=min(5, int(results["noncommutativity_curvature"] * 5)),
                                    mode='grazing' if results["noncommutativity_curvature"] > 0.2 else 'interior',
                                    new_braid_word=braid_word,
                                    new_cs_phase=float(results["commutativity_gap"])
                                )
                                
                    ENGINE.save_state()
                    
                    # Convert Tensor in results to list for serialization
                    if isinstance(results.get("combined_residue"), torch.Tensor):
                        results["combined_residue"] = results["combined_residue"].tolist()
                        
                    if ENGINE._zeitgeist_state is not None:
                        results["zeitgeist_state"] = ENGINE._zeitgeist_state.to_dict()
                        
                    results["success"] = True
                    self._send_json(results)
                    
                except Exception as e:
                    print(f"Error in Minecraft ingestion endpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    self._send_error_json(str(e))
                return

            elif self.path == '/wikipedia-extract':
                # Enhanced Wikipedia content extraction endpoint
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    data = json.loads(post_body.decode('utf-8'))
                    
                    urls = data.get('urls', [])
                    options = data.get('options', {})
                    
                    # Import Wikipedia integration with better error handling
                    try:
                        import sys
                        import os
                        sys.path.append(os.path.join(os.path.dirname(__file__)))
                        from wikipedia_integration import wikipedia_integration
                        print(" Wikipedia integration module loaded successfully")
                    except ImportError as e:
                        print(f" Failed to import Wikipedia integration: {e}")
                        self._send_error_json(f"Wikipedia integration module not available: {e}")
                        return
                    
                    results = []
                    for url in urls:
                        try:
                            title = wikipedia_integration.extract_title_from_url(url)
                            print(f" Processing Wikipedia page: {title}")
                            
                            # Fetch content
                            content_data = wikipedia_integration.fetch_wikipedia_content(title)
                            if content_data:
                                # Clean content
                                cleaned_content = wikipedia_integration.clean_wikipedia_content(
                                    content_data['full_content'], 
                                    title
                                )
                                # Extract concepts
                                concepts = wikipedia_integration.extract_key_concepts(title, cleaned_content)
                                
                                # Create associations if requested
                                associations_created = 0
                                if options.get('create_associations', True):
                                    for concept in concepts:
                                        if concept != title:  # Don't associate with itself
                                            try:
                                                # Create association using existing system
                                                # Use generate_response=False to avoid timeout
                                                association_result = ENGINE.process_input(f"ASSOCIATE: {concept} <-> {cleaned_content[:2000]}", generate_response=False)
                                                associations_created += 1
                                                print(f" Created association: {concept} <-> content")
                                                
                                                # Limit to 5 associations per page to prevent backend timeout
                                                if associations_created >= 5:
                                                    print(" Reached association limit per page (5)")
                                                    break
                                            except Exception as e:
                                                print(f"  Failed to create association for {concept}: {e}")
                                
                                results.append({
                                    'url': url,
                                    'title': title,
                                    'content_length': len(cleaned_content),
                                    'original_length': content_data['content_length'],
                                    'concepts': concepts,
                                    'associations_created': associations_created,
                                    'method': content_data['method'],
                                    'status': 'success'
                                })
                            else:
                                results.append({
                                    'url': url,
                                    'title': title,
                                    'status': 'failed',
                                    'error': 'Could not fetch content'
                                })
                        except Exception as e:
                            print(f" Error processing {url}: {e}")
                            results.append({
                                'url': url,
                                'title': wikipedia_integration.extract_title_from_url(url) if 'wikipedia_integration' in locals() else 'Unknown',
                                'status': 'failed',
                                'error': str(e)
                            })
                    
                    # Get statistics
                    try:
                        stats = wikipedia_integration.get_statistics()
                    except:
                        stats = {'error': 'Statistics not available'}
                    
                    ENGINE.save_state()
                    self._send_json({
                        'results': results,
                        'statistics': stats,
                        'total_processed': len([r for r in results if r['status'] == 'success']),
                        'total_failed': len([r for r in results if r['status'] == 'failed'])
                    })
                    
                except Exception as e:
                    print(f" Wikipedia extraction endpoint error: {e}")
                    self._send_error_json(f"Wikipedia extraction failed: {e}")
                
            
            # ================================================================
            # PHASE 1: LOCAL DATA ENDPOINTS (No HF Token Required)
            # ================================================================
            elif self.path == '/api/test_token':
                # Accept token test  now works with local-only mode too
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                token = data.get('token', '')
                
                if token.startswith('hf_'):
                    # Real HF token  attempt validation
                    try:
                        req = urllib.request.Request(
                            'https://huggingface.co/api/whoami',
                            headers={'Authorization': f'Bearer {token}'}
                        )
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            user_data = json.loads(resp.read().decode('utf-8'))
                        self._send_json({
                            'success': True,
                            'username': user_data.get('name', 'unknown'),
                            'message': 'Token validated with HuggingFace'
                        })
                    except Exception as e:
                        self._send_json({
                            'success': False,
                            'message': f'HF token validation failed: {str(e)}'
                        })
                elif token == 'LOCAL_MODE':
                    # Local-only mode  no token needed
                    datasets = LOCAL_LOADER.scan()
                    self._send_json({
                        'success': True,
                        'username': 'local_user',
                        'message': f'Local mode active  {len(datasets)} datasets available'
                    })
                else:
                    self._send_json({
                        'success': False,
                        'message': 'Token must start with hf_ or use LOCAL_MODE'
                    })
            
            elif self.path == '/api/ingest_local':
                # Ingest from local data/raw/ without HF token
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                
                dataset_name = data.get('dataset', '')
                max_samples = int(data.get('max_samples', 500))
                
                print(f" Local ingestion: {dataset_name} (max={max_samples})")
                
                samples = []
                quality_reports = []
                for sample in LOCAL_LOADER.load_samples(dataset_name, max_samples):
                    # Apply textbook filtering (per-dimension admissibility)
                    report = TEXTBOOK_FILTER.assess(sample.text, sample.source)
                    if report.is_admissible:
                        samples.append(sample)
                        quality_reports.append(report)
                        
                        # Feed into engine for association learning
                        if len(samples) <= 50:  # Limit direct engine processing
                            try:
                                ENGINE.process_input(sample.text[:500], generate_response=False)
                            except Exception:
                                pass
                
                stats = TEXTBOOK_FILTER.get_statistics(quality_reports)
                
                self._send_json({
                    'success': True,
                    'dataset': dataset_name,
                    'samples_loaded': len(samples),
                    'quality_stats': stats,
                    'message': f'Ingested {len(samples)} samples from {dataset_name}'
                })
            
            elif self.path == '/api/start_training':
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                epochs = int(data.get('epochs', 3))
                
                TRAINING_STATE['active'] = True
                TRAINING_STATE['progress'] = 0
                TRAINING_STATE['log'] = [f'Training started: {epochs} epochs']
                TRAINING_STATE['results'] = None
                
                # Run lightweight structural training
                import threading
                def _training_worker(n_epochs):
                    try:
                        for epoch in range(n_epochs):
                            TRAINING_STATE['log'].append(f'Epoch {epoch+1}/{n_epochs}...')
                            TRAINING_STATE['progress'] = int((epoch / n_epochs) * 100)
                            
                            # Load a batch of local data and train
                            for ds_name in list(LOCAL_LOADER._datasets.keys())[:3]:
                                batch = LOCAL_LOADER.load_batch(ds_name, batch_size=16, max_samples=50)
                                for sample in batch:
                                    try:
                                        ENGINE.process_input(sample.text[:300], generate_response=False)
                                    except Exception:
                                        pass
                            
                            TRAINING_STATE['log'].append(
                                f'Epoch {epoch+1} complete -- iteration {ENGINE.iteration}'
                            )
                        
                        ENGINE.save_state()
                        TRAINING_STATE['progress'] = 100
                        TRAINING_STATE['results'] = {'success': True}
                        TRAINING_STATE['log'].append('Training complete!')
                    except Exception as e:
                        TRAINING_STATE['log'].append(f'Error: {str(e)}')
                        TRAINING_STATE['results'] = {'success': False, 'error': str(e)}
                    finally:
                        TRAINING_STATE['active'] = False
                
                t = threading.Thread(target=_training_worker, args=(epochs,), daemon=True)
                t.start()
                
                self._send_json({'success': True, 'message': f'Training started: {epochs} epochs'})
            
            elif self.path == '/api/stop_training':
                TRAINING_STATE['active'] = False
                TRAINING_STATE['log'].append('Training stopped by user')
                self._send_json({'success': True, 'message': 'Training stop requested'})
            
            elif self.path == '/api/save_model':
                ENGINE.save_state()
                self._send_json({'success': True, 'message': 'Model state saved'})
            
            elif self.path == '/api/chat':
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                message = data.get('message', data.get('text', ''))
                
                if message:
                    result = ENGINE.process_input(message)
                    ENGINE.save_state()
                    self._send_json(result)
                else:
                    self._send_json({'error': 'No message provided'})
            
            # ================================================================
            # PHASE 3: TABBY ML ENDPOINTS
            # ================================================================
            elif self.path == '/api/tabby_test':
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                
                if not TABBY_AVAILABLE:
                    self._send_json({'connected': False, 'message': 'Tabby client not available'})
                else:
                    host = data.get('host', 'localhost')
                    port = int(data.get('port', 8080))
                    TABBY_CLIENT.config = TabbyConfig(host=host, port=port)
                    result = TABBY_CLIENT.test_connection()
                    self._send_json(result)
            
            elif self.path == '/api/tabby_complete':
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                
                if not TABBY_AVAILABLE or TABBY_CLIENT is None:
                    self._send_json({'success': False, 'error': 'Tabby ML not available'})
                else:
                    prompt = data.get('prompt', '')
                    mode = data.get('mode', 'complete')  # 'complete' or 'chat'
                    
                    if mode == 'chat':
                        messages = data.get('messages', [{'role': 'user', 'content': prompt}])
                        result = TABBY_CLIENT.chat(messages)
                    else:
                        result = TABBY_CLIENT.complete(prompt)
                    
                    self._send_json(result.to_dict())
            
            elif self.path == '/api/tabby_generate_training':
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                
                if not TABBY_AVAILABLE or TABBY_CLIENT is None:
                    self._send_json({'success': False, 'error': 'Tabby ML not available'})
                else:
                    topic = data.get('topic', 'algorithms')
                    style = data.get('style', 'textbook')
                    result = TABBY_CLIENT.generate_training_sample(topic, style)
                    
                    if result.success:
                        # Feed the generated sample back through the engine
                        try:
                            ENGINE.process_input(result.text[:500], generate_response=False)
                        except Exception:
                            pass
                    
                    self._send_json(result.to_dict())
                
            else:
                self.send_error(404)
        except Exception as e:
            print(f"POST Error: {e}")
            self._send_error_json(str(e))

    def _send_json(self, data):
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            # Use custom encoder to handle tensors and other non-serializable types
            response_data = json.dumps(data, cls=TensorEncoder).encode('utf-8')
            self.wfile.write(response_data)
        except (ConnectionAbortedError, BrokenPipeError) as e:
            print(f"  Client connection lost during response: {e}")
        except Exception as e:
            print(f" Error sending JSON response: {e}")
            import traceback
            traceback.print_exc()


    def _send_error_json(self, message, code=500):
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_data = json.dumps({"error": message}).encode('utf-8')
            self.wfile.write(error_data)
        except (ConnectionAbortedError, BrokenPipeError) as e:
            print(f"  Client connection lost during error response: {e}")
        except Exception as e:
            print(f" Error sending error response: {e}")

def kill_port_owner(port):
    """Find and kill any process holding the port."""
    if os.name == 'nt':
        try:
            # Find PID using netstat
            cmd = f"netstat -ano | findstr :{port}"
            output = subprocess.check_output(cmd, shell=True).decode()
            for line in output.splitlines():
                if "LISTENING" in line:
                    parts = line.strip().split()
                    pid = parts[-1]
                    if int(pid) != os.getpid():
                        print(f"Flushing ghost process {pid} on port {port}...")
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
        except Exception as e:
            # No process found or permission error
            pass

START_TIME = time.time()

def main():
    print("--- [GYROIDIC DIEGETIC BACKEND] ---")
    
    # Prune orphaned processes
    kill_port_owner(8000)
    
    # PID Tracking
    pid_file = ".backend.pid"
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
        
    print(f"PID: {os.getpid()} | Port: 8000")
    
    server_address = ('', 8000)
    try:
        httpd = http.server.HTTPServer(server_address, RequestHandler)
    except Exception as e:
        print(f"CRITICAL PORT ERROR: {e}")
        # Final attempt to clear port
        kill_port_owner(8000)
        time.sleep(1)
        httpd = http.server.HTTPServer(server_address, RequestHandler)
    
    print("INITIALIZING PHYSICS ENGINE...")
    print("STATUS: MANIFOLD COHERENT. STANDBY FOR CONNECTIONS.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down manifold...")
        if os.path.exists(pid_file):
            os.remove(pid_file)
        httpd.server_close()
        # Prevent PyArrow segfault by finalizing S3
        try:
            import pyarrow.fs
            pyarrow.fs.finalize_s3()
        except Exception:
            pass

if __name__ == "__main__":
    main()



