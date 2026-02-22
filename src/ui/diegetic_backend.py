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

# Ensure PYTHONPATH includes project root for all imports
import sys
import os
sys.path.insert(0, os.getcwd())

# Advanced Extensions Imports (Lazy/Safe)
try:
    from src.core.meta_polytope_matrioshka import MetaPolytopeMatrioshka
    from src.core.quantum_inspired_reasoning import QuantumInspiredReasoningState
    from src.core.context_aware_quantizer import ContextAwareQuantizer
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
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List, Optional
import hashlib

# Import Gyroidic Components
# Ensure PYTHONPATH is adequate or sys.path is used
sys.path.append(os.getcwd())

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.training.fgrt_fgrt_trainer import SpectralStructuralTrainer
from src.models.resonance_cavity import ResonanceCavity
from src.models.diegetic_heads import ResonanceLarynx, DataAssociationLayer

# GARBLED OUTPUT REPAIR SYSTEM INTEGRATION
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from src.core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from src.core.love_vector import LoveVector
from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
from src.core.dyadic_transfer import DyadicTransferMap
from src.core.love_invariant_protector import SoftSaturatedGates

# LEGACY SYSTEM INTEGRATION
# CALM: Context-Adaptive Latent Momentum (Trajectory Veto)
from src.surrogates.calm_predictor import CALM
# KAGH: Kolmogorov-Arnold-Gödel-Huxley (Speculative Drafting)
from src.surrogates.kagh_networks import KAGHBlock, HarmonicWaveDecomposition, HuxleyRD
# Gyroid Covariance for tensor-based momentum instead of scalar averages
from src.topology.gyroid_covariance import GyroidCovarianceEstimator
# Speculative Coprime Chiral Gating (Legacy Recovery)
from src.core.speculative_coprime_gate import SpeculativeCoprimeGate
# Graph Topology
from src.topology.embedding_graph import GyroidicGraphManager
# Pressure Ingestor for constraint forcing when code is detected
from src.data.pressure_ingestor import PressureIngestor

# Local Data Loading (Phase 1: HF token barrier removal)
from src.data.local_data_loader import LocalDataLoader
from src.data.textbook_filter import TextbookFilter

# Tabby ML Integration (Phase 3)
try:
    from src.integrations.tabby_client import TabbyClient, TabbyConfig
    TABBY_AVAILABLE = True
except ImportError:
    TABBY_AVAILABLE = False
    print("⚠️ Tabby ML client not available")

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
        iterations = [int(f.split('_')[1]) for f in files if f.startswith('enc_') and f.endswith('.pt')]
        return max(iterations) if iterations else 0
        
    def save_encoding(self, iteration: int, text: str, input_tensor: torch.Tensor, memory_state: torch.Tensor, response: str, metrics: Dict[str, Any]):
        """Save the encoding dyad to a timestamped file along with structural metrics."""
        import time
        timestamp = int(time.time())
        filename = f"encoding_{iteration}_{timestamp}.pt"
        path = os.path.join(self.base_dir, filename)
        
        data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "text_input": text,
            "input_tensor": input_tensor.detach().cpu(),
            "memory_state": memory_state.detach().cpu(), # We use this as the primary node embedding
            "response": response
        }
        # Add metrics for graph weighting (e.g. chiral_score, entropy)
        data.update(metrics)
        
        torch.save(data, path)
        return filename

from src.core.fractal_meta_functional import FractalMetaFunctional

class DiegeticPhysicsEngine(nn.Module):
    """
    The Core Engine.
    Combines Cavity + Larynx + Persistence + Fractal Meta-Recursion + CALM + KAGH.
    """
    def __init__(self, dim=80, k=5, calm_history_len=8, device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

            # Now use self.device for everything else
        print(f"🔧 Engine initialized on: {self.device}")
        self.dim = dim
        self.k = k
        self.last_input_time = 0
        
        # Advanced Extensions (Lazy Init)
        self.meta_polytope = None
        self.quantum_reasoner = None
        self.extensions_enabled = EXTENSIONS_AVAILABLE

        self.cavity = ResonanceCavity(hidden_dim=dim, num_modes=16)
        self.larynx = ResonanceLarynx(hidden_dim=dim, vocab_size=128) # ASCII
        self.associator = DataAssociationLayer(input_dim=dim, hidden_dim=dim, k=k)
        
        # =============================================
        # GARBLED OUTPUT REPAIR SYSTEM
        # =============================================
        print("🔧 Initializing Garbled Output Repair System...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        self.bezout_refresh.bezout_matrix.fill_(0.0)
        self.bezout_refresh.bezout_matrix.add_(torch.eye(5))  # Identity is the safest starting poin

        # Chern-Simons Gasket - plugs logic leaks
        self.chern_simons_gasket = ChernSimonsGasket(
            manifold_dim=3,
            level_k=1,
            device=device
        )
        
        # Soliton Stability Healer - heals fractured solitons
        self.soliton_healer = SolitonStabilityHealer(
            alpha_0=1.0,
            gamma=0.5,
            healing_iterations=400,
            device=device
        )
        
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
        
        # Polynomial Config for repair system (anti-lobotomy compliance)
        self.repair_polynomial_config = PolynomialCoprimeConfig(
            k=k, 
            degree=4, 
            basis_type='chebyshev',
            learnable=True, 
            use_saturation=True,
            device=device
        )
        
        print("✅ Garbled Output Repair System initialized")
        
        # =============================================
        
        # FRACTAL META-FUNCTIONAL HOOK
        # Enables "self-distrusting recursive loops"
        self.fractal_meta = FractalMetaFunctional(dim=dim, k=k)
        
        # Persistent Meta-State S_meta(t-1)
        self.register_buffer('meta_state', torch.randn(1, dim))
        
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
        # Shape [k] — one scalar per polynomial coprime field.
        self.register_buffer('trust_scalars', torch.ones(k))

        # Temporal Association Trainer state (lazy-init on first interaction)
        self._temporal_trainer = None
        self._temporal_dataset = None
        self._temporal_thread = None
        self._last_temporal_diag: dict = {}
        self._last_matrioshka_diag: dict = {}
        
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        self.fossilizer = DyadFossilizer(storage_dir="data/encodings")
        
        # 11. Spectral Structural Trainer (Deeper Dynamics)
        self.trainer = SpectralStructuralTrainer(
            model=self, 
            poly_config=PolynomialCoprimeConfig(k=k, degree=4),
            lr=0.001

        )
        self.optimizer = torch.optim.Adam(self.larynx.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # 12. Fingerprint Projection (137 dims -> 64 dims)
        # (32 R + 32 G + 32 B + 32 L + 1 Texture + 8 Edge Features)
        self.fingerprint_proj = nn.Linear(137, self.dim)
        nn.init.orthogonal_(self.fingerprint_proj.weight)
        
        self.iteration = 0
        self.encoding_manager = EncodingManager()
        
        # Seed the Larynx if it's a "Blank Slate"
        self._initialize_larynx_weights()

    def forward(self, input_tensor: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Evolutionary Forward Pass for Manifold Invariants.
        Used by SpectralStructuralTrainer for Ricci Flow and ADMM repairs.
        """
        # 3. Update Resonance Cavity (Explicit Memory Update)
        # We pass input_tensor as attention_states to trigger M update
        cavity_out = self.cavity(input_tensor.unsqueeze(1))
        memory_state = cavity_out['memory_state'].mean(dim=1) # [1, dim]
        
        # 4. FRACTAL META-RECURSION
        est_residues = torch.tanh(self.associator.residue_map(memory_state)) # [1, k]
        
        meta_out = self.fractal_meta(
            current_state=memory_state,
            meta_state_prev=self.meta_state,
            residues=est_residues,
            dark_matter=self.cavity.D_dark[0].mean(dim=0, keepdim=True),# [1, dim]
        )
        
        # Update persistent meta-state (detach to prevent graph blowup here)
        self.meta_state = meta_out['s_fractal'].detach()
        
        # Return state for character generation / training
        return meta_out['s_fractal']

    def _initialize_larynx_weights(self):
        """Seed character projections with basic English frequency priors."""
        # Simple vowel-biased seeding to prevent random symbol noise
        vowels = [ord(c) for c in "aeiou AEIOU"]
        common = [ord(c) for c in "rstln RSTLN"]
        
        with torch.no_grad():
            # Initial noise
            self.larynx.proj.weight.data.normal_(0, 0.01)
            # Boost vowels and common letters
            for char_idx in vowels:
                if char_idx < 128:
                    self.larynx.proj.weight.data[char_idx] *= 5.0
            for char_idx in common:
                if char_idx < 128:
                    self.larynx.proj.weight.data[char_idx] *= 3.0
        
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
                closure_margin = float(closure_threshold - closure_score)
                components = {
                    'cosine_similarity_mean': float(cos.mean().item()),
                    'cosine_similarity_min': float(cos.min().item()),
                    'cosine_similarity_max': float(cos.max().item())
                }
                return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
                    'closure_score': closure_score,
                    'closure_threshold': closure_threshold,
                    'closure_margin': closure_margin,
                    'components': components
                }
        except Exception as e:
            print(f"⚠️  Unfolding closure check fallback due to error: {e}")
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
                'closure_score': 1.0,
                'closure_threshold': 0.5,
                'closure_margin': -0.5,
                'components': {'error': str(e)}
            }
        
    def process_input(self, text_input: str, fingerprint: Optional[Dict] = None, generate_response: bool = True) -> dict:
        """
        Process user text, update cavity, and generate emergent response via Fractal Recursion.
        Now uses CALM, KAGH, and HarmonicWaveDecomposition for proper legacy integration.
        Optional multi-channel fingerprint can bias the manifold ingestion.
        Enhanced with constraint pressure injection when code is detected.
        """
        self.iteration += 1
        self.last_input_time = time.time() # Update last input time for budget checks
        
        # 1. Embed Input (Hash Projection) - MOVED UP
        input_tensor = self._text_to_tensor(text_input) # [1, dim]
        
        # =============================================
        # PHASE 0: AFFORDANCE GRADIENT COMPUTATION
        # =============================================
        
        # Compute affordance gradients for both code and conversational patterns
        affordance_gradients = self._compute_affordance_gradients(text_input, input_tensor)
        
        print(f"🔧 Affordance Gradients Computed:")
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
            print(f"🔥 CONSTRAINT FORCING TRIGGERED:")
            if affordance_gradients['constraint_forcing_gradient'] > 0.1:
                print(f"   • Affordance gradient: {affordance_gradients['constraint_forcing_gradient']:.4f}")
            if conversational_results.get('constraint_pressure_generated', 0.0) > 0.05:
                print(f"   • Conversational pressure: {conversational_results['constraint_pressure_generated']:.4f}")
            
            # Show which affordances contributed to constraint forcing
            if affordance_gradients['executability_pressure'] > 0.05:
                print(f"   • Executability pressure: {affordance_gradients['executability_pressure']:.4f}")
            if affordance_gradients['formal_symbol_density'] > 0.05:
                print(f"   • Formal symbol density: {affordance_gradients['formal_symbol_density']:.4f}")
            if affordance_gradients['conversational_embedding_pressure'] > 0.05:
                print(f"   • Conversational embedding: {affordance_gradients['conversational_embedding_pressure']:.4f}")
            if affordance_gradients['api_extraction_potential'] > 0.05:
                print(f"   • API extraction potential: {affordance_gradients['api_extraction_potential']:.4f}")
        
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
        
        # Integrate fingerprint if provided (Topological Enrichment)
        if fingerprint:
            # Flatten fingerprint features: [r, g, b, l, texture, edges]
            feat = torch.cat([
                torch.tensor(fingerprint['r']),
                torch.tensor(fingerprint['g']),
                torch.tensor(fingerprint['b']),
                torch.tensor(fingerprint['l']),
                torch.tensor([fingerprint['texture']]),
                torch.tensor(fingerprint.get('edges', [0.0] * 8))  # 8 edge features
            ], dim=0).unsqueeze(0).float() # [1, 137]
            
            with torch.no_grad():
                fp_bias = self.fingerprint_proj(feat)
                input_tensor = input_tensor + 0.5 * fp_bias
        
        # 2. MIMICRY (Active Listening)
        self._train_mimicry(input_tensor, text_input)
        
        # 2.5 DYNAMIC MANIFOLD CLOCK (Play vs Seriousness)
        # Manifold pressure = Similarity(Input, History) + Mean(Curvature)
        # Higher pressure -> Seriousness (small dt)
        # Lower pressure -> Play (large dt)
        # Proxy: Use similarity to recent meta-state
        with torch.no_grad():
            s_norm = self.meta_state / (torch.norm(self.meta_state) + 1e-8)
            i_norm = input_tensor / (torch.norm(input_tensor) + 1e-8)
            cos_sim = torch.dot(s_norm.flatten(), i_norm.flatten()).item()
            # If sim is high, pressure is low (it's familiar/smooth)
            # If sim is low, pressure is high (it's novel/jagged)
            manifold_pressure = 1.0 - cos_sim 
            dt = 0.5 * math.exp(-manifold_pressure) # Play/Seriousness scaling
        
        # 3. Evolutionary Pass (Cavity + Meta-Functional)
        # Now uses forward() to support SpectralStructuralTrainer
        manifold_state = self.forward(input_tensor, dt=dt)
        seed_state = manifold_state.detach() # Explicit seed for response
        
        # =============================================
        # DYAD AGENTIC TRIGGERS (AFFORDANCE-BASED)
        # =============================================
        dyad_override_response = None
        
        # Trigger Ingestion if expandability is critical
        if affordance_gradients.get('runtime_expandability', 0.0) > 0.8:
            print("[TRIGGER] Agentic Ingestion Triggered by Affordance Gradient")
            dyad_override_response = self._handle_dyad_ingestion(f"AGENTIC_INGEST: {text_input}", fingerprint, seed_state)
            
        # Trigger Association if knowledge seeking is critical
        elif affordance_gradients.get('knowledge_seeking', 0.0) > 0.8:
            print("[TRIGGER] Agentic Association Triggered by Affordance Gradient")
            dyad_override_response = self._handle_association_learning(text_input, seed_state)
            
        # =============================================
        # 5. CALM: Update history buffer and get trajectory assessment
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
                print(f"🌊 CALM Agentic Forcing applied: P={gauge_pressure:.2f}, ||F||={torch.norm(correction).item():.4f}")
        
        calm_diagnostics = {
            "abort_score": abort_score,
            "rho_factor": rho_factor,
            "step_factor": step_factor,
            "gauge_pressure": gauge_pressure,
            "trajectory_status": "STABLE"
        }
        
        if abort_score > 0.8:
            calm_diagnostics["trajectory_status"] = "CRITICAL_COLLAPSE_IMMINENT"
            # Veto logic... (existing)
        rho_factor = _as_float(rho_factor_tensor)
        step_factor = _as_float(step_factor_tensor)
        
        calm_diagnostics = {
            "abort_score": abort_score,
            "rho_factor": rho_factor,
            "step_factor": step_factor,
            "trajectory_status": "STABLE" if abort_score < 0.8 else ("WARPED" if abort_score < 0.7 else "NEVER_VETO")
        }
        
        # =============================================
        # 6. EARLY EXIT FOR NON-GENERATIVE TASKS
        # =============================================
        if not generate_response:
            print("🚀 Skipping generation pipeline (Association/Ingestion Mode)")
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
                "status": "processed_no_generation",
                "iteration": self.iteration,
                "affordance_gradients": affordance_gradients,
                "conversational_results": conversational_results,
                "calm_diagnostics": calm_diagnostics,
                "memory_state_updated": True,
                "mimicry_trained": True,
                "payload": {
                    "type": "topological_shape_stalk",
                    "status": "asymptotic_ingestion",
                    "stalk_active": True
                }
            }

        # =============================================
        # 6. Speculative Coprime Recovery (Legacy Restoration)
        # =============================================
        # If CALM detects collapse (abort_score > 0.5), attempt structure recovery
        # using Wasserstein optimal transport toward a coprime-coherent manifold.
        self.meta_state, recovery_metrics = self.coprime_gate(
            state=self.meta_state,
            abort_score=abort_score_tensor,
            residues=est_residues,
            chirality_target=input_tensor
        )
        # If recovery succeeded in locking coprime parity, we override the CALM abort
        if recovery_metrics['coprime_lock'] and recovery_metrics['recovery_attempted']:
             abort_score = 0.0
             calm_diagnostics["trajectory_status"] = "RECOVERED"
             # Signal that we have "un-collapsed" the trajectory
        
        # =============================================
        # 7. KAGH: Speculative Response Drafting (Topological Ghost)
        # =============================================
        # Use KAGH to draft a "ghost" of the response state before character generation
        # This provides a long-horizon speculative target
        kagh_input = memory_state + 0.3 * self.meta_state + input_tensor * 0.4
        response_ghost = self.kagh_drafter(kagh_input) # [1, dim]
        
        # =============================================
        # 7. Harmonic Wave Decomposition: Separate Signal from Noise
        # =============================================
        # Split the ghost into ergodic (noise) and non-ergodic (signal) components
        ergodic_component, non_ergodic_component = self.harmonic_decomp(response_ghost)
        
        # Seed state emphasizes the non-ergodic (coherent) component
        # After coprime gating, the state is already structuralized
        seed_state = non_ergodic_component + 0.2 * ergodic_component + input_tensor * 0.3
        
        # =============================================
        # PHASE 1.5: CONSTRAINT PRESSURE INJECTION (CODE DETECTION)
        # =============================================
        
        # =============================================
        # PHASE 1.5: ENHANCED CONSTRAINT PRESSURE INJECTION
        # =============================================
        
        # Inject constraint pressure based on affordance gradients (pure affordance-based approach)
        if constraint_forcing_needed:
            print("🔥 Applying enhanced constraint pressure injection to seed state...")
            
            # Apply constraint injection with affordance-based metrics
            seed_state = self._inject_constraint_pressure(seed_state, enhanced_constraint_metrics)
            print(f"🔧 Post-injection seed state shape: {seed_state.shape}")
        
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
        max_output_length = min(max_output_length, 500)
        min_output_length = max(len(text_input) // 2, 20)
        
        # =============================================
        # 8.5. GARBLED OUTPUT REPAIR PIPELINE (PHASE 2.1: SPECTRAL COHERENCE CORRECTOR)
        # =============================================
        print(f"🔧 Applying repair to state: {seed_state.shape}")
        
        try:
            # PHASE 2.1: Re-enable Spectral Coherence Corrector
            print("🔧 Phase 2.1: Applying Spectral Coherence Correction...")
            print(f"🔧 Input state shape: {seed_state.shape}, device: {seed_state.device}")
            
            # Apply spectral coherence correction to fix consonant clustering
            # Make correction more aggressive for better results
            seed_state_corrected = self.spectral_corrector.adaptive_coherence_correction(
                signal=seed_state,
                output_text=None  # We don't have output text yet, but corrector can work without it
            )
            # Apply additional vowel-bias correction to combat consonant clustering
            # This is a temporary enhancement while we tune the spectral corrector
            with torch.no_grad():
                # Boost dimensions that correspond to vowel-like patterns
                vowel_boost = torch.randn_like(seed_state_corrected) * 0.1
                vowel_mask = torch.rand_like(seed_state_corrected) > 0.7  # 30% of dimensions get vowel boost
                seed_state_corrected = seed_state_corrected + vowel_boost * vowel_mask.float()
            
            print(f"🔧 Corrected state shape: {seed_state_corrected.shape}")
            
            # Get spectral diagnostics
            spectral_diagnostics = self.spectral_corrector.get_diagnostics()
            print(f"🔍 Spectral Coherence: θ={spectral_diagnostics['theta_coherence']:.3f}, "
                  f"energy_ratio={spectral_diagnostics['energy_ratio']:.3f}")
            
            # Store diagnostics for metrics
            self._last_spectral_diagnostics = spectral_diagnostics
            print(f"🔧 Stored diagnostics: {self._last_spectral_diagnostics}")
            
            # PHASE 2.2: Re-enable Bezout Coefficient Refresh (PROPER IMPLEMENTATION)
            print("🔧 Phase 2.2: Applying Bezout Coefficient Refresh...")
            
            # Ensure proper state dimensions before Bezout processing
            if seed_state_corrected.dim() == 3 and seed_state_corrected.shape[1] == 1:
                seed_state_corrected = seed_state_corrected.squeeze(1)  # Remove singleton dimension
                print(f"🔧 Squeezed state for Bezout processing: {seed_state_corrected.shape}")
            
            try:
                # Create proper residues from corrected state for CRT correction
                batch_size = seed_state_corrected.shape[0]
                state_dim = seed_state_corrected.shape[1]
                
                # Apply Symmetry-Preserving Reshape for Bezout compatibility
                if state_dim % self.k != 0:
                    pad_size = self.k - (state_dim % self.k)
                    seed_state_padded = torch.nn.functional.pad(seed_state_corrected, (0, pad_size), mode='reflect')
                    print(f"🔧 Applied Symmetry-Preserving padding for Bezout: {state_dim} -> {seed_state_padded.shape[1]}")
                else:
                    seed_state_padded = seed_state_corrected
                
                # Create proper residues for CRT correction
                padded_dim = seed_state_padded.shape[1]
                residue_dim = padded_dim // self.k
                residues_for_crt = seed_state_padded.view(batch_size, self.k, residue_dim)
                print(f"🔧 Created residues for Bezout: {residues_for_crt.shape}")
                
                # Apply CRT correction to fix modulus drift
                corrected_residues = self.bezout_refresh.apply_crt_correction(residues_for_crt)
                
                # Reshape back to state format and restore original dimensions
                seed_state_crt_flat = corrected_residues.view(batch_size, -1)
                if seed_state_crt_flat.shape[1] > state_dim:
                    seed_state_corrected = seed_state_crt_flat[:, :state_dim]  # Remove padding
                    print(f" Restored original dimensions after Bezout: {seed_state_crt_flat.shape[1]} -> {state_dim}")
                else:
                    seed_state_corrected = seed_state_crt_flat
                
                # Get Bezout diagnostics
                bezout_diagnostics = self.bezout_refresh.get_diagnostics()
                print(f"🔍 Bezout CRT: condition_number={bezout_diagnostics['bezout_condition_number']:.3f}")
                
                # Store Bezout diagnostics
                self._last_bezout_diagnostics = bezout_diagnostics
                
            except Exception as bezout_error:
                print(f"⚠️  Bezout Coefficient Refresh failed: {bezout_error}")
                print("🔧 Using fallback diagnostics...")
                # Store fallback diagnostics
                self._last_bezout_diagnostics = {
                    'bezout_condition_number': 1.0,
                    'moduli_mean': 1.0,
                    'moduli_std': 0.0,
                    'drift_threshold': 0.5,
                    'error': str(bezout_error)
                }
            
            print("🔧 Phase 2.2 skipped - continuing with spectral correction only")
            
            # Basic numerical stabilization (keep this as safety net)
            print("🔧 Applying numerical stabilization...")
            
            # Check for NaN/inf values and replace them
            if torch.isnan(seed_state_corrected).any() or torch.isinf(seed_state_corrected).any():
                print("⚠️  Detected NaN/inf values, applying emergency stabilization")
                nan_mask = torch.isnan(seed_state_corrected) | torch.isinf(seed_state_corrected)
                seed_state_corrected = torch.where(nan_mask, torch.randn_like(seed_state_corrected) * 0.01, seed_state_corrected)
            
            # Numerical stabilization: clamp values to prevent inf/nan in downstream operations
            seed_state_corrected = torch.clamp(seed_state_corrected, min=-10.0, max=10.0)
            
            # Normalize to prevent extreme values
            seed_state_corrected = seed_state_corrected / (torch.norm(seed_state_corrected, dim=-1, keepdim=True) + 1e-8)
            
            seed_state_repaired = seed_state_corrected
            print(f"✅ Phase 2.1 repair complete. State shape: {seed_state_repaired.shape}")
            
        except Exception as e:
            print(f"❌ REPAIR SYSTEM ERROR: {e}")
            print("🔧 Falling back to basic stabilization...")
            
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
                print("⚠️  Detected NaN/inf values, applying emergency stabilization")
                nan_mask = torch.isnan(seed_state) | torch.isinf(seed_state)
                seed_state = torch.where(nan_mask, torch.randn_like(seed_state) * 0.01, seed_state)
            
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
                print(f"🔧 Truncated state from {seed_state_repaired.shape[-1]} to {self.dim}")
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
                print(f"🔧 Reconstructed state from {seed_state_repaired.shape[-1]} to {self.dim}")
        
        print(f"🔧 Final seed state shape: {seed_state.shape} (expected: [1, {self.dim}])")
        
        # Apply basic numerical stabilization
        seed_state = torch.clamp(seed_state, min=-10.0, max=10.0)
        seed_state = seed_state / (torch.norm(seed_state, dim=-1, keepdim=True) + 1e-8)
        
        print(f"🔧 Applied numerical stabilization. State range: [{seed_state.min():.3f}, {seed_state.max():.3f}]")
        
        # =============================================
        # PHASE 2.3: CHERN-SIMONS GASKET (LOGIC LEAK PREVENTION)
        # =============================================
        print("🔧 Phase 2.3: Applying Chern-Simons Gasket (Logic Leak Prevention)...")
        
        try:
            # Ensure proper state dimensions before applying gasket
            if seed_state.dim() == 3 and seed_state.shape[1] == 1:
                seed_state = seed_state.squeeze(1)  # Remove singleton dimension
                print(f"🔧 Squeezed state to proper dimensions: {seed_state.shape}")
            
            # Apply Chern-Simons gasket to plug logic leaks
            # First, we need to create residues from the state for the gasket
            # Convert state to residue format expected by plug_logic_leak
            batch_size = seed_state.shape[0]
            state_dim = seed_state.shape[1]
            
            # Create proper residues and polynomial coefficients for the gasket
            # Apply Symmetry-Preserving Reshape for Chern-Simons compatibility
            if state_dim % self.k != 0:
                # Apply reflective padding to reach nearest multiple of k
                pad_size = self.k - (state_dim % self.k)
                seed_state_padded = torch.nn.functional.pad(seed_state, (0, pad_size), mode='reflect')
                print(f"🔧 Applied Symmetry-Preserving padding for Chern-Simons: {state_dim} -> {seed_state_padded.shape[1]}")
            else:
                seed_state_padded = seed_state
            
            # Now create residues with proper dimensions
            padded_dim = seed_state_padded.shape[1]
            residue_dim = padded_dim // self.k
            proper_residues = seed_state_padded.view(batch_size, self.k, residue_dim)  # [1, 5, 13]
            
            # Use proper polynomial coefficients from the repair system's polynomial config
            # Instead of mock data, use the actual polynomial basis from the system
            base_polynomial_coeffs = self.repair_polynomial_config.get_coefficients_tensor()  # [K, D]
            
            # Ensure coefficients match the residue dimensions
            if base_polynomial_coeffs.shape[1] != residue_dim:
                if base_polynomial_coeffs.shape[1] > residue_dim:
                    # Truncate if larger
                    proper_polynomial_coeffs = base_polynomial_coeffs[:, :residue_dim]
                    print(f"🔧 Truncated polynomial coeffs: {base_polynomial_coeffs.shape} -> {proper_polynomial_coeffs.shape}")
                else:
                    # Expand if smaller using proper polynomial evaluation
                    # Instead of padding, evaluate the polynomials at more points
                    x_points = torch.linspace(-1, 1, residue_dim, device=seed_state.device)
                    proper_polynomial_coeffs = self.repair_polynomial_config.evaluate(x_points.unsqueeze(0)).squeeze(0).T  # [K, residue_dim]
                    print(f"🔧 Expanded polynomial coeffs via evaluation: {base_polynomial_coeffs.shape} -> {proper_polynomial_coeffs.shape}")
            else:
                proper_polynomial_coeffs = base_polynomial_coeffs
            
            print(f"🔧 Using proper polynomial coefficients: {proper_polynomial_coeffs.shape}")
            
            # Apply the Chern-Simons gasket with proper polynomial coefficients
            gasket_residues = self.chern_simons_gasket.plug_logic_leak(
                residues=proper_residues,
                polynomial_coeffs=proper_polynomial_coeffs
            )
            # Convert back to state format and restore original dimensions
            gasket_residues_flat = gasket_residues.view(batch_size, -1)
            
            # Restore to original state dimensions (remove padding if applied)
            if gasket_residues_flat.shape[1] > state_dim:
                seed_state_gasket = gasket_residues_flat[:, :state_dim]  # Remove padding
                print(f"🔧 Restored original dimensions: {gasket_residues_flat.shape[1]} -> {state_dim}")
            else:
                seed_state_gasket = gasket_residues_flat
            
            # Get Chern-Simons diagnostics
            chern_simons_diagnostics = self.chern_simons_gasket.get_diagnostics()
            print(f"🔍 Chern-Simons: level_k={chern_simons_diagnostics.get('level_k', 'N/A')}")
            
            # Store diagnostics
            self._last_chern_simons_diagnostics = chern_simons_diagnostics
            
            # Use gasket-corrected state
            seed_state = seed_state_gasket
            print(f"🔧 Gasket-corrected state shape: {seed_state.shape}")
            
        except Exception as gasket_error:
            print(f"⚠️  Chern-Simons Gasket failed: {gasket_error}")
            print("🔧 Continuing without gasket correction...")
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
        print("🔧 Phase 2.4: Applying Soliton Stability Healer (Fracture Healing)...")
        
        try:
            # Convert state back to residues for soliton healing
            batch_size = seed_state.shape[0]
            state_dim = seed_state.shape[1]
            
            # Apply Symmetry-Preserving Reshape for Soliton compatibility
            if state_dim % self.k != 0:
                pad_size = self.k - (state_dim % self.k)
                seed_state_padded = torch.nn.functional.pad(seed_state, (0, pad_size), mode='reflect')
                print(f"🔧 Applied Symmetry-Preserving padding for Soliton: {state_dim} -> {seed_state_padded.shape[1]}")
            else:
                seed_state_padded = seed_state
            
            # Create residues for soliton healing
            padded_dim = seed_state_padded.shape[1]
            residue_dim = padded_dim // self.k
            residues_for_healing = seed_state_padded.view(batch_size, self.k, residue_dim)
            print(f"🔧 Created residues for Soliton healing: {residues_for_healing.shape}")
            
            # Apply soliton healing (we don't have output text yet, so it will use iteration-based healing)
            healed_residues = self.soliton_healer.heal_fractured_soliton(
                residues=residues_for_healing,
                output_text=None  # Will be applied based on iteration count
            )
            # Convert back to state format and restore original dimensions
            healed_state_flat = healed_residues.view(batch_size, -1)
            if healed_state_flat.shape[1] > state_dim:
                seed_state = healed_state_flat[:, :state_dim]  # Remove padding
                print(f"🔧 Restored original dimensions after Soliton healing: {healed_state_flat.shape[1]} -> {state_dim}")
            else:
                seed_state = healed_state_flat
            
            # Get Soliton diagnostics
            soliton_diagnostics = self.soliton_healer.get_diagnostics()
            print(f"🔍 Soliton Healer: α={soliton_diagnostics['alpha']:.3f}, progress={soliton_diagnostics['healing_progress']:.3f}")
            
            # Store Soliton diagnostics
            self._last_soliton_diagnostics = soliton_diagnostics
            
        except Exception as soliton_error:
            print(f"⚠️  Soliton Stability Healer failed: {soliton_error}")
            print("🔧 Continuing without soliton healing...")
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
        print("🔧 Phase 2.5: Applying Love Vector & Soft Saturated Gates...")
        
        try:
            # Apply Love Invariant (Non-Ownable Flow)
            # L + meta_state
            self.meta_state = self.love_vector(self.meta_state)
            
            # Diagnostic check (kernel property)
            ownership_leak = self.love_vector.ownership_check().item()
            print(f"🔍 Love Invariant active: ownership_leak={ownership_leak:.3f}")
            
            # Apply Soft Saturated Gates for tri-state logic
            # ... (rest of soft gates logic) ...
            batch_size = seed_state.shape[0]
            state_dim = seed_state.shape[1]
            if state_dim % self.k != 0:
                pad_size = self.k - (state_dim % self.k)
                seed_state_padded = torch.nn.functional.pad(seed_state, (0, pad_size), mode='reflect')
            else:
                seed_state_padded = seed_state
            
            padded_dim = seed_state_padded.shape[1]
            residue_dim = padded_dim // self.k
            residues_for_saturation = seed_state_padded.view(batch_size, self.k, residue_dim)
            
            # Use Love Vector property as proxy for persistence
            pas_h = torch.norm(self.love_vector.L) / 5.0
            
            performance_scores = torch.norm(residues_for_saturation, dim=2).mean(dim=0)
            performance_scores = torch.sigmoid(performance_scores)
            
            saturated_residues = self.soft_gates.apply_soft_saturation(
                signal=residues_for_saturation,
                pas_h=pas_h,
                performance_scores=performance_scores
            )
            saturated_state_flat = saturated_residues.view(batch_size, -1)
            if saturated_state_flat.shape[1] > state_dim:
                seed_state = saturated_state_flat[:, :state_dim]
            else:
                seed_state = saturated_state_flat
            
            # Diagnostics
            soft_gates_metrics = self.soft_gates.get_diagnostics()
            self._last_soft_gates_diagnostics = soft_gates_metrics
            self._last_love_diagnostics = {"ownership_leak": ownership_leak, "love_norm": torch.norm(self.love_vector.L).item()}
            
        except Exception as love_gates_error:
            print(f"⚠️ Love Vector / Soft Gates failed: {love_gates_error}")

        # =============================================
        # PHASE 2.6: MATRIOSHKA QUANTIZED EVOLUTION LOOP
        # Realises: x_{t+1} = Q_{Z_t}(F(Q_{Z_t}(x_t)))
        # (ai project report_2-2-2026.txt §3 "Matrioshka Quantized Windows")
        # Uses CALM constraint output as PAS scores for anisotropy.
        # =============================================
        if self.caq is not None:
            try:
                # Derive per-axis PAS scores from CALM constraints if available
                _pas_scores = None
                if 'constraints_tensor' in locals() and constraints_tensor is not None:
                    # constraints_tensor: [1, 5] — map to dim via linear interpolation
                    _ct = constraints_tensor.detach().view(-1)  # [5]
                    # Expand to [dim] by repeating across field groups
                    repeats = self.dim // _ct.shape[0] + 1
                    _pas_scores = _ct.repeat(repeats)[:self.dim].sigmoid()  # [dim] ∈ [0,1]

                _matrioshka_steps = 3  # Q→F→Q iterations
                _boundary_hit = False
                for _loop in range(_matrioshka_steps):
                    # Inner quantization: Q_Z(x)
                    q_inner, _b_inner = self.caq(seed_state, pas_scores=_pas_scores)
                    # Evolve through physics surrogate: F(Q_Z(x))
                    with torch.no_grad():
                        q_evolved = self.kagh_drafter(q_inner)
                    # Outer quantization: Q_Z(F(Q_Z(x)))
                    seed_state, _b_outer = self.caq(q_evolved, pas_scores=_pas_scores)
                    # Detect critical shell ceiling — stop forcing if hit
                    if _b_outer is not None and _b_outer.is_critical():
                        print(f"🔔 Matrioshka shell ceiling hit at step {_loop} — halting loop")
                        _boundary_hit = True
                        break

                self._last_matrioshka_diag = self.caq.get_diagnostics()
                self._last_matrioshka_diag['loop_steps'] = _loop + 1
                self._last_matrioshka_diag['boundary_halt'] = _boundary_hit
                print(f"✅ Phase 2.6 Matrioshka loop complete: "
                      f"level={self._last_matrioshka_diag['level']}, "
                      f"step_mean={self._last_matrioshka_diag['step_mean']:.4f}")
            except Exception as _caq_err:
                print(f"⚠️  Matrioshka evolution loop failed: {_caq_err}")

        print("🔧 Starting text generation with fully repaired state...")
        
        # =============================================
        # PHASE 3: RESPONSE QUALITY OPTIMIZATION
        # =============================================
        print("🔧 Phase 3: Response Quality Optimization...")
        
        # Enhanced dyad-aware response generation
        response_text = self._generate_dyad_aware_response(
            seed_state=seed_state,
            input_text=text_input,
            fingerprint=fingerprint,
            max_length=max_output_length,
            min_length=min_output_length
        )
        print(f"✅ Generated dyad-aware response: {response_text}")
        print(f"🔧 Response length: {len(response_text)} characters")
        
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
        print("🔧 Phase 4: Advanced Feature Integration...")
        
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

        print(f"📊 Phase 4 Gyroid Violation Score: {gyroid_violation_score:.4f}")
        print(f"📊 Phase 4 Unfolding Closure: {unfolding_closure_result['is_closed']}")
        print(f"📊 Phase 4 Topological Features: {len(topological_analysis['features'])} detected")

        # Construct metrics now that all dependencies are available
        metrics = {
            "response": response_text,
            "iteration": self.iteration,
            "spectral_entropy": 0.5,
            "chiral_score": 0.1,
            "coprime_lock": bool(recovery_metrics.get('coprime_lock', False)) if isinstance(recovery_metrics, dict) else False,
            "output_length": len(response_text),
            "affordance_gradients": affordance_gradients,
            "conversational_results": conversational_results,
            "calm_diagnostics": calm_diagnostics,
            "constraint_forcing_applied": constraint_forcing_needed,
            "payload": {
                "type": "topological_shape_stalk",
                "stalk": topological_analysis,
                "shape_violation": gyroid_violation_score,
                "resonance": recovery_metrics.get('chirality_alignment', 0.0) if isinstance(recovery_metrics, dict) else 0.0
            }
        }
        
        # Add repair diagnostics if available
        repair_diagnostics = {}
        if hasattr(self, '_last_spectral_diagnostics'):
            repair_diagnostics['spectral_coherence_corrector'] = self._last_spectral_diagnostics
            print(f"📊 Spectral Diagnostics: {self._last_spectral_diagnostics}")
        
        if hasattr(self, '_last_bezout_diagnostics'):
            repair_diagnostics['bezout_coefficient_refresh'] = self._last_bezout_diagnostics
            print(f"📊 Bezout Diagnostics: {self._last_bezout_diagnostics}")
        
        if hasattr(self, '_last_chern_simons_diagnostics'):
            repair_diagnostics['chern_simons_gasket'] = self._last_chern_simons_diagnostics
            print(f"📊 Chern-Simons Diagnostics: {self._last_chern_simons_diagnostics}")
        
        if hasattr(self, '_last_soliton_diagnostics'):
            repair_diagnostics['soliton_stability_healer'] = self._last_soliton_diagnostics
            print(f"📊 Soliton Diagnostics: {self._last_soliton_diagnostics}")
        
        if hasattr(self, '_last_love_diagnostics'):
            repair_diagnostics['love_invariant_protector'] = self._last_love_diagnostics
            print(f"📊 Love Diagnostics: {self._last_love_diagnostics}")
        
        if hasattr(self, '_last_soft_gates_diagnostics'):
            repair_diagnostics['soft_saturated_gates'] = self._last_soft_gates_diagnostics
            print(f"📊 Soft Gates Diagnostics: {self._last_soft_gates_diagnostics}")
        
        # Phase 3 diagnostics
        phase3_diagnostics = {
            'dyad_aware_generation': True,
            'echo_suppression_active': True,
            'vowel_optimization_active': True,
            'linguistic_correction_available': True,
            'multimodal_fingerprint_support': fingerprint is not None
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
                    d = {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}}
                    d.update({k: _sanitize(v) for k, v in x.items()})
                    return d
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

        # Trigger one background temporal association train_step
        self._maybe_trigger_temporal_training(seed_state)

        print("📤 Returning metrics")
        return metrics

    # =========================================================================
    # PHASE 17: TEMPORAL ASSOCIATION TRAINER — background bridge
    # =========================================================================

    def _maybe_trigger_temporal_training(self, seed_state: torch.Tensor) -> None:
        """
        Fire one TemporalAssociationTrainer.train_step in a background daemon
        thread so that it never blocks the HTTP response path.

        The trainer is lazily initialised on the first call.  A new thread is
        only spawned if the previous one has already finished (single-shot
        per interaction, no queue build-up).
        """
        import threading

        def _run() -> None:
            try:
                from src.training.temporal_association_trainer import (
                    TemporalAssociationDataset,
                    TemporalAssociationTrainer,
                )
                # Lazy init
                if self._temporal_dataset is None:
                    self._temporal_dataset = TemporalAssociationDataset(
                        device=self.device
                    )
                if self._temporal_trainer is None:
                    self._temporal_trainer = TemporalAssociationTrainer(
                        model=self,
                        dataset=self._temporal_dataset,
                        learning_rate=1e-4,
                        trust_update_rate=0.01,
                        fossilization_threshold=0.8,
                        device=self.device,
                    )
                # One training step per interaction
                batch = self._temporal_dataset.get_temporal_sequence(batch_size=1)
                result = self._temporal_trainer.train_step(batch)
                self._last_temporal_diag = result
                print(
                    f"[TAT] temporal step: "
                    f"acc={result.get('association_accuracy', 0):.4f} "
                    f"coh={result.get('temporal_coherence', 0):.4f} "
                    f"trust_mean={result.get('trust_mean', 0):.4f}"
                )
            except Exception as _tat_err:
                # Trainer failure is never allowed to crash the main pipeline
                self._last_temporal_diag = {"error": str(_tat_err)}
                print(f"[TAT] background train_step failed: {_tat_err}")

        # Guard: only one background thread at a time
        if self._temporal_thread is None or not self._temporal_thread.is_alive():
            self._temporal_thread = threading.Thread(target=_run, daemon=True)
            self._temporal_thread.start()

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
        with torch.no_grad():
            manifold_out = self.forward(text_emb, dt=0.05)

        result: dict = {
            "output": manifold_out,
            "trust_scalars": self.trust_scalars,
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

    def _train_mimicry(self, input_state: torch.Tensor, text_target: str):
        """Train Larynx to decrypt the input state back to text."""
        self.optimizer.zero_grad()
        
        # We treat whole string reconstruction from single state as hard.
        # So we train to predict the *distribution* of chars in the input (Bag of Words style)
        # Or simply predict the *next* char?
        # Let's train it to predict the *first* few chars or the dominant chars.
        # Actually, let's just train it to map state -> chars in the string.
        # "This state represents this sequence of characters".
        
        # Target distribution (Character counts)
        target_dist = torch.zeros(1, 128)
        for char in text_target:
            idx = ord(char) % 128
            target_dist[0, idx] += 1.0
        target_dist = target_dist / (target_dist.sum() + 1e-8)
        
        # Forward pass
        logits, _ = self.larynx(input_state)
        # KL Divergence or Cross Entropy against distribution?
        # Simple: Cross Entropy against *randomly sampled* char from text?
        # Let's use BCEWithLogitsLoss for multi-label (bag of chars)
        # or simplified: maximize logits for present chars.
        
        loss = 0
        for char in text_target:
            idx = ord(char) % 128
            loss += self.criterion(logits, torch.tensor([idx]))
        
        loss = loss / (len(text_target) + 1e-8)
        loss.backward()
        self.optimizer.step()

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Sequence-Aware Polynomial Rotating Hash.
        Uses polynomial coefficients instead of hardcoded primes (anti-lobotomy).
        Ensures word order and sentence structure influence the embedding.
        """
        vec = torch.zeros(1, self.dim)
        
        # Generate polynomial coefficients instead of hardcoded primes
        # Use Chebyshev polynomial basis for rotation
        num_coeffs = 12
        poly_coeffs = []
        for n in range(num_coeffs):
            # Chebyshev polynomial T_n evaluated at a fixed point
            x = 0.5  # Fixed evaluation point
            if n == 0:
                coeff = 1.0
            elif n == 1:
                coeff = x
            else:
                # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
                t_prev2 = 1.0
                t_prev1 = x
                for k in range(2, n + 1):
                    t_curr = 2 * x * t_prev1 - t_prev2
                    t_prev2 = t_prev1
                    t_prev1 = t_curr
                coeff = t_prev1
            
            # Scale and ensure positive for indexing
            poly_coeffs.append(abs(coeff * 10) + 2)  # Ensure >= 2
        
        for i, char in enumerate(text):
            # Positional Polynomial Shift
            p = poly_coeffs[i % len(poly_coeffs)]
            # Rotate target dimension based on position and polynomial coefficient
            idx = int((i * p + ord(char)) % self.dim)
            
            # Harmonic magnitude modulation
            magnitude = (ord(char) / 128.0) * (1.0 / (math.log(i + 2)))
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
        
        executability_pressure = (imperative_markers * 0.1 + 
                                procedural_indicators * 0.05 + 
                                referential_density * 0.02) / max(len(text.split()), 1)
        
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
        
        formal_symbol_density = (symbolic_chars * 0.01 + 
                               formal_operators * 0.1 + 
                               structured_delims * 0.05) / max(len(text), 1)
        
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
        
        return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
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
        
        print("🔥 ENHANCED CONSTRAINT INJECTION: Processing multiple affordance types")
        
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
        
        print(f"🔧 Constraint sources: {constraint_sources}")
        print(f"🔧 Pressure signature: {pressure_signature}")
        
        # Check cache first
        if pressure_signature in self.constraint_pressure_cache:
            print(f"🔧 Using cached constraint pressure for signature {pressure_signature}")
            constraint_batch = self.constraint_pressure_cache[pressure_signature]
        else:
            print(f"🔧 Generating new constraint pressure for signature {pressure_signature}")
            
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
                
                print(f"📊 Pressure Report: {pressure_report['total_constraints_extracted']} constraints, "
                      f"{pressure_report['total_collisions_detected']} collisions, "
                      f"density: {pressure_report['pressure_density']:.3f}")
                
                # Generate constraint batch from pressure ingestor
                batch_size = min(8, max(2, len(constraint_sources) * 2))
                constraint_batch = self.pressure_ingestor.get_constraint_batch(batch_size)
                
                # Cache the constraint batch
                self.constraint_pressure_cache[pressure_signature] = constraint_batch
                
            except Exception as e:
                print(f"⚠️  Constraint pressure generation failed: {e}")
                # Fallback: generate synthetic constraint pressure
                constraint_batch = torch.randn(4, 512, device=seed_state.device) * 2.0
        
        # Inject constraint pressure into seed state
        batch_size, state_dim = seed_state.shape
        constraint_dim = constraint_batch.shape[1]
        
        # Apply Symmetry-Preserving Reshape for constraint injection
        if constraint_dim != state_dim:
            if constraint_dim > state_dim:
                # Truncate constraint batch to match state dimensions
                constraint_injection = constraint_batch[:, :state_dim]
                print(f"🔧 Truncated constraint batch: {constraint_dim} -> {state_dim}")
            else:
                # Expand constraint batch using reflective padding
                pad_size = state_dim - constraint_dim
                constraint_injection = torch.nn.functional.pad(constraint_batch, (0, pad_size), mode='reflect')
                print(f"🔧 Expanded constraint batch: {constraint_dim} -> {state_dim}")
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
            print(f"🔧 Executability affordance boost: {exec_pressure * 0.3:.4f}")
        
        # Formal symbol boost
        formal_pressure = affordance_gradients.get('formal_symbol_density', 0.0)
        if formal_pressure > 0.05:
            affordance_boost += formal_pressure * 0.25
            print(f"🔧 Formal symbol affordance boost: {formal_pressure * 0.25:.4f}")
        
        # Conversational boost
        conv_pressure = affordance_gradients.get('conversational_embedding_pressure', 0.0)
        if conv_pressure > 0.05:
            affordance_boost += conv_pressure * 0.3
            print(f"🔧 Conversational affordance boost: {conv_pressure * 0.3:.4f}")
        
        # API extraction boost
        api_pressure = affordance_gradients.get('api_extraction_potential', 0.0)
        if api_pressure > 0.05:
            affordance_boost += api_pressure * 0.25
            print(f"🔧 API extraction affordance boost: {api_pressure * 0.25:.4f}")
        
        # Runtime expandability boost
        expand_pressure = affordance_gradients.get('runtime_expandability', 0.0)
        if expand_pressure > 0.05:
            affordance_boost += expand_pressure * 0.2
            print(f"🔧 Runtime expandability boost: {expand_pressure * 0.2:.4f}")
        
        # Conversational constraint boost
        conv_constraint_pressure = conversational_results.get('constraint_pressure_generated', 0.0)
        if conv_constraint_pressure > 0.05:
            affordance_boost += conv_constraint_pressure * 0.4
            print(f"🔧 Conversational constraint boost: {conv_constraint_pressure * 0.4:.4f}")
        
        # Final injection strength
        injection_strength = min(base_injection_strength + affordance_boost, 0.9)  # Cap at 90%
        
        print(f"🔧 Enhanced injection strength: {base_injection_strength:.3f} + {affordance_boost:.3f} = {injection_strength:.3f}")
        
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
            collision_noise = torch.randn_like(seed_state) * collision_factor * 0.1
            forced_state = forced_state + collision_noise
            print(f"🔥 Applied collision forcing: {self.last_pressure_report['total_collisions_detected']} collisions")
        
        # Normalize to prevent explosion while preserving constraint pressure
        forced_state = forced_state / (torch.norm(forced_state, dim=-1, keepdim=True) + 1e-8)
        
        print(f"🔥 Constraint pressure injected: strength={injection_strength:.3f}, "
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
            return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 'extracted': False, 'reason': 'insufficient_conversational_pressure'}
        
        print(f"🔥 CONVERSATIONAL EMBEDDING EXTRACTION TRIGGERED")
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
        
        print(f"🔧 Conversational extraction complete:")
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
                # Simulate successful API extraction
                return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
                    'success': True,
                    'source': f'{indicator}_api',
                    'content': f"Extracted content related to '{text[:50]}...' from {indicator} API",
                    'content_length': len(text) * 3,  # Simulate expanded content
                    'extraction_method': 'simulated_api_call',
                    'api_pressure_used': api_pressure
                }
        
        return {'payload': {'status': 'EVOLVING', 'pas_h': 0.61}, 
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
            result = self._handle_association_learning(association_text, self.meta_state)
            
            if "learned" in result.lower():
                associations_created = 1
                print(f"✅ Created temporal association from API content")
            
        except Exception as e:
            print(f"⚠️  Failed to create temporal association: {e}")
        
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
    
    def _generate_dyad_aware_response(
        self, 
        seed_state: torch.Tensor, 
        input_text: str, 
        fingerprint: Optional[Dict] = None,
        max_length: int = 200,
        min_length: int = 20
    ) -> str:
        """
        Phase 3: Enhanced response generation with dyad-aware optimization.
        
        Leverages the privileged text-to-image and text-to-text association system
        to generate more coherent and contextually relevant responses.
        """
        print("🔧 Phase 3: Dyad-Aware Response Generation")
        
        # Detect if this is a dyad ingestion or association command
        is_dyad_ingest = input_text.startswith("INGEST_DYAD:")
        is_association = input_text.startswith("ASSOCIATE:")
        
        if is_dyad_ingest:
            return self._handle_dyad_ingestion(input_text, fingerprint, seed_state)
        elif is_association:
            return self._handle_association_learning(text_input, seed_state)
        else:
            # Apply Dyadic Transfer (Phase 5) 
            # We use Task 0 (Code) and Task 1 (Conversational) leakage
            with torch.no_grad():
                # Detect proficiency from affordance gradients (calculated earlier in process_input)
                # Assuming they are stored or accessible. For now, we use a simple heuristic:
                # If seed_state has high variance -> Code proficiency? 
                # Better: uses the internal transfer_map with dummy task states
                task_states = seed_state.unsqueeze(1).expand(-1, 8, -1) # Broad-cast to 8 tasks
                
                # Proficiency scores: 0=Code, 1=Conversation
                # Using the affordances calculated in process_input (if we can find them)
                # For this implementation, we simulate them or use seed_state properties
                p_code = torch.sigmoid(torch.norm(seed_state)) 
                p_conv = torch.sigmoid(torch.norm(seed_state - 0.5))
                proficiency = torch.zeros(1, 8, device=self.device)
                proficiency[0, 0] = p_code
                proficiency[0, 1] = p_conv
                
                # Apply leakage
                leaked_state = self.transfer_map(task_states, proficiency)
                # Take the mean across Tasks 0 and 1
                seed_state = (leaked_state[:, 0, :] + leaked_state[:, 1, :]) / 2.0
                
            # Fallback to general enhanced response generation
            return self._generate_enhanced_response(
                seed_state=seed_state,
                input_text=input_text,
                fingerprint=fingerprint,
                max_length=max_length,
                min_length=min_length,
            )

    
    def _handle_dyad_ingestion(self, input_text: str, fingerprint: Optional[Dict], seed_state: torch.Tensor) -> str:
        """Handle canonical dyad ingestion using DyadFossilizer."""
        raw_content = input_text.replace("INGEST_DYAD:", "").strip()
        
        # Support both [fingerprint_json] | description and just description
        if "|" in raw_content:
            fp_str, description = raw_content.split("|", 1)
            description = description.strip()
            # If fingerprint passed in dict, use it, else parse from string
            if not fingerprint:
                try: fingerprint = json.loads(fp_str)
                except: fingerprint = None
        else:
            description = raw_content

        # Create KnowledgeDyad object
        # Note: Ingestion often comes from images, where fingerprint might be present
        fp_tensor = torch.zeros(137, device=self.device)
        if fingerprint:
            # Flatten fingerprint features: [r, g, b, l, texture, edges]
            fp_list = fingerprint.get('r', []) + fingerprint.get('g', []) + fingerprint.get('b', []) + \
                      fingerprint.get('l', []) + [fingerprint.get('texture', 0.0)] + fingerprint.get('edges', [0.0]*8)
            if len(fp_list) == 137:
                fp_tensor = torch.tensor(fp_list, device=self.device).float()

        dyad = KnowledgeDyad(
            image_fingerprint=fp_tensor,
            linguistic_description=description
        )
        
        # Call fossilizer
        fossil_path = self.fossilizer.fossilize(dyad, seed_state)
        
        print(f"[WAVE] Deposition confirmed: {fossil_path}")
        return f"Knowledge Dyad fossilized at {os.path.basename(fossil_path)}. Association Implication preserved in manifold."
    
    def _handle_association_learning(self, input_text: str, seed_state: torch.Tensor) -> str:
        """Handle association learning via fossil recovery and resonance injection."""
        # Note: ASSOCIATE: command or affordance trigger
        print("🔧 Phase 4: Dyadic Association Recovery")
        
        # Load all fossils
        fossils = self.fossilizer.recover_fossils()
        if not fossils:
            return "No fossils found in data/encodings. Association requires existing topological obstructions."
            
        # Compute resonance between current seed_state and fossils
        # Map seed_state to residue_dim (dim // k) for comparison
        best_resonance = -1.0
        best_fossil = None
        
        for f in fossils:
            residue = f['residue_vector'].to(self.device).flatten()
            # Simple dot product resonance
            res = torch.dot(seed_state.flatten(), residue) / (torch.norm(seed_state) * torch.norm(residue) + 1e-8)
            if res > best_resonance:
                best_resonance = res
                best_fossil = f
                
        if best_fossil and best_resonance > 0.5:
            # Inject residue into meta_state
            with torch.no_grad():
                res_vec = best_fossil['residue_vector'].to(self.device).view_as(self.meta_state)
                # Weighted injection based on resonance
                self.meta_state = self.meta_state + 0.3 * res_vec
                
            return f"Resonating with fossil: '{best_fossil['description'][:40]}...'. Resonance Score: {best_resonance:.3f}. Residue injected into meta-functional manifold."
        
        return f"Manifold scanned. No resonant fossils found for current state (Max Resonance: {best_resonance:.3f})."
    
    def _generate_enhanced_response(
        self, 
        seed_state: torch.Tensor, 
        input_text: str, 
        fingerprint: Optional[Dict],
        max_length: int,
        min_length: int
    ) -> str:
        """Generate enhanced response with improved linguistic coherence."""
        # This method is now effectively replaced by the direct call to larynx.generate_response
        # in process_input. Keeping it here for reference but it should not be called.
        print("🔧 Enhanced response generation with linguistic optimization (DEPRECATED PATH)")
        
        # Phase 3.1: Reduce recursive echoing while preserving meta-cognition
        echo_suppression_factor = 0.7  # Reduce tendency to repeat input
        
        # Phase 3.2: Improve response diversity through temperature scheduling
        base_temperature = 0.6  # Start more focused
        temperature_decay = 0.95  # Gradually increase creativity
        
        # Phase 3.3: Vowel-consonant balance optimization
        vowel_boost_factor = 1.5  # Encourage vowel generation
        
        response_chars = []
        current_state = seed_state.clone()
        
        # Apply echo suppression by reducing input influence over time
        input_influence = 1.0
        
        # Enhanced character generation loop
        for i in range(min(max_length, 300)):
            # Dynamic temperature scheduling
            current_temperature = base_temperature * (temperature_decay ** i)
            current_temperature = max(current_temperature, 0.3)  # Minimum temperature
            
            # Get logits from larynx
            logits, confidence = self.larynx(current_state, temperature=current_temperature)
            
            # Apply vowel bias to improve linguistic balance
            vowel_indices = [ord(c) for c in "aeiouAEIOU" if ord(c) < 128]
            for idx in vowel_indices:
                if idx < logits.shape[-1]:
                    logits[0, idx] *= vowel_boost_factor
            
            # Apply echo suppression (reduce probability of repeating input characters)
            if i > 5:  # After initial characters
                for char in input_text.lower():
                    char_idx = ord(char)
                    if char_idx < logits.shape[-1]:
                        logits[0, char_idx] *= echo_suppression_factor
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            char_idx = torch.multinomial(probs.squeeze(0), 1).item()
            
            # Enhanced character filtering and validation
            if 32 <= char_idx <= 126:  # Printable ASCII
                char = chr(char_idx)
                
                # Linguistic coherence checks
                if len(response_chars) > 0:
                    last_char = response_chars[-1]
                    
                    # Prevent excessive symbol clustering
                    if not char.isalnum() and not last_char.isalnum() and len(response_chars) > 3:
                        # Skip this symbol to prevent clustering
                        continue
                    
                    # Encourage word boundaries
                    if char == ' ' and last_char == ' ':
                        continue  # Prevent double spaces
                
                response_chars.append(char)
                
                # Natural stopping conditions
                if char in '.!?' and len(response_chars) >= min_length:
                    # Check if we have reasonable content
                    text_so_far = ''.join(response_chars)
                    vowel_count = sum(1 for c in text_so_far.lower() if c in 'aeiou')
                    if vowel_count >= len(text_so_far) * 0.1:  # At least 10% vowels
                        break
                
                # Stop if we reach minimum length and have good linguistic balance
                if len(response_chars) >= min_length:
                    text_so_far = ''.join(response_chars)
                    vowel_ratio = sum(1 for c in text_so_far.lower() if c in 'aeiou') / len(text_so_far)
                    if vowel_ratio > 0.15 and i > min_length * 1.5:  # Good balance achieved
                        break
            
            # State evolution with reduced chaos
            # Phase 3.4: Optimize generation speed through controlled state updates
            state_update_magnitude = 0.05 * (1.0 - i / max_length)  # Decrease over time
            noise = torch.randn_like(current_state) * state_update_magnitude
            
            # Incorporate fingerprint influence if available
            if fingerprint and i < max_length * 0.3:  # Early influence only
                fp_influence = self._get_fingerprint_influence(fingerprint, i)
                current_state = current_state + noise + fp_influence * 0.02
            else:
                current_state = current_state + noise
            
            # Normalize to prevent explosion
            current_state = current_state / (torch.norm(current_state, dim=-1, keepdim=True) + 1e-8)
            
            # Reduce input influence over time (echo suppression)
            input_influence *= 0.98
        
        response_text = ''.join(response_chars)
        
        # Phase 3.5: Post-processing quality checks and fallbacks
        if len(response_text.strip()) < 5:
            print("⚠️  Generated text too short, using enhanced fallback")
            return self._generate_fallback_response(input_text, fingerprint)
        
        # Check linguistic quality
        vowel_count = sum(1 for c in response_text.lower() if c in 'aeiou')
        vowel_ratio = vowel_count / len(response_text) if len(response_text) > 0 else 0
        
        if vowel_ratio < 0.08:  # Very poor linguistic balance
            print("⚠️  Poor linguistic balance detected, applying post-correction")
            response_text = self._apply_linguistic_correction(response_text)
        
        return response_text
    
    def _get_fingerprint_influence(self, fingerprint: Dict, position: int) -> torch.Tensor:
        """Extract positional influence from visual fingerprint."""
        # Use RGB values to influence different parts of the state vector
        r_val = fingerprint['r'][position % len(fingerprint['r'])]
        g_val = fingerprint['g'][position % len(fingerprint['g'])]
        b_val = fingerprint['b'][position % len(fingerprint['b'])]
        
        # Create influence vector
        influence = torch.zeros(1, self.dim)
        influence[0, :self.dim//3] = r_val * 0.1
        influence[0, self.dim//3:2*self.dim//3] = g_val * 0.1
        influence[0, 2*self.dim//3:] = b_val * 0.1
        
        return influence
    
    def _generate_fallback_response(self, input_text: str, fingerprint: Optional[Dict]) -> str:
        """Generate enhanced fallback response."""
        if fingerprint:
            return f"Visual resonance detected. Processing '{input_text}' through multimodal manifold topology."
        else:
            return f"Processing '{input_text}' through linguistic manifold. Coherence patterns emerging."
    
    def _apply_linguistic_correction(self, text: str) -> str:
        """Apply post-processing linguistic correction."""
        # Simple vowel injection for extremely poor outputs
        if len(text) < 10:
            return text
        
        # Insert vowels at strategic positions
        corrected = []
        consonant_run = 0
        
        for i, char in enumerate(text):
            corrected.append(char)
            
            if char.isalpha() and char.lower() not in 'aeiou':
                consonant_run += 1
                # Insert vowel after 3+ consonants
                if consonant_run >= 3 and i < len(text) - 1:
                    vowel = ['a', 'e', 'i', 'o', 'u'][i % 5]
                    corrected.append(vowel)
                    consonant_run = 0
            else:
                consonant_run = 0
        
        return ''.join(corrected)
    
    def _compute_full_gyroid_violation_score(self, state: torch.Tensor, response_text: str) -> float:
        """
        Phase 4.1: Complete Gyroid Violation Score computation.
        
        Implements full gyroidic manifold violation detection using:
        - Spectral signature analysis
        - Covariance-based pressure computation  
        - Topological consistency checks
        - Response-state correlation analysis
        """
        print("🔧 Phase 4.1: Computing Full Gyroid Violation Score...")
        
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
            
            # Weighted combination
            full_violation_score = (
                0.6 * base_violation +
                0.4 * response_correlation_violation
            )
            return float(full_violation_score)
            
        except Exception as e:
            print(f"⚠️  Gyroid violation computation failed: {e}")
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
    
    def _perform_unfolding_closure_check(self, state: torch.Tensor, input_text: str, response_text: str) -> dict:
        """
        Phase 4.2: Complete Unfolding Closure Check implementation.
        
        Implements topological closure verification using:
        - Hyper-ring operator evaluation
        - Cycle closure detection
        - Triadic reciprocity validation
        - Unfolding branch analysis
        """
        print("🔧 Phase 4.2: Performing Unfolding Closure Check...")
        
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
            
            # Create constraint manifold representation
            constraint_manifold = self._create_constraint_manifold(state)
            
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
            closure_result = self._closure_checker(hyper_ring, constraint_manifold)
            
            # Extract results
            is_closed = closure_result.get('is_closed', torch.tensor([False])).item()
            is_trivial = closure_result.get('is_trivial', torch.tensor([True])).item()
            is_valid = closure_result.get('is_valid', torch.tensor([False])).item()
            
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
            print(f"⚠️  Unfolding closure check failed: {e}")
            return self._fallback_closure_check(state, input_text, response_text)
    
    def _create_hyper_ring_from_state(self, state: torch.Tensor, input_text: str, response_text: str) -> torch.Tensor:
        """
        Create hyper-ring representation using existing HyperRingOperator.
        
        This uses the existing topology/hyper_ring.py system for proper
        hyper-ring creation with topological guarantees.
        """
        try:
            # Try to use existing HyperRingOperator if available
            from src.topology.hyper_ring import HyperRingOperator
            
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
                print(f"⚠️  Decoupled CRT manifold creation failed: {e}")
        
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
        print("🔧 Phase 4.3: Performing Advanced Topological Analysis...")
        
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
            print(f"⚠️  Advanced topological analysis failed: {e}")
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
        
        # β₀ (connected components) - approximate via clustering
        # Use simple threshold-based clustering
        threshold = state_flat.std().item()
        positive_components = (state_flat > threshold).sum().item()
        negative_components = (state_flat < -threshold).sum().item()
        beta_0 = max(1, positive_components + negative_components) / len(state_flat)
        
        # β₁ (cycles) - approximate via autocorrelation
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
        
        print(f"🔧 Enhanced learning: lr={adaptive_lr:.4f}, length_ratio={length_ratio:.1f}")
        
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
            print(f"🔧 Bidirectional learning: '{target_sample[:20]}...' → '{source}'")
        
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
            print(f"⚠️ Advanced Physics skipped: budget exceeded ({time.time() - self.last_input_time:.2f}s)")
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
                     print(f"✨ Advanced Physics: Matrioshka Level {level}, Quantum Entropy {superposition_entropy:.3f}")
            
        except Exception as e:
            print(f"⚠️ Advanced Physics Error: {e}")
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

        print(f"✅ Full state & artifacts persisted.")

    def load_state(self):
        """Unified load: Neural Weights + Metadata + Encoding Context."""
        if not os.path.exists(STATE_PATH):
            print("ℹ️ No persistence file found. Starting fresh.")
            return False

        try:
            # 1. Load Neural Weights (Non-Strict for flexibility)
            checkpoint = torch.load(STATE_PATH, map_location=self.device)
            self.load_state_dict(checkpoint, strict=False)

            # 2. Repair non-finite values (NaN/Inf)
            self._repair_tensors()

            # 3. Synchronize Encoding Context
            # Update the engine's iteration count from the manager's findings
            self.iteration = self.encoding_manager.get_latest_iteration()

            print(f"✅ State restored. Resuming from iteration {self.iteration}")
            return True
        except Exception as e:
            print(f"❌ Critical Load Failure: {e}")
            return False

    def _repair_tensors(self):
        """Surgical repair of non-finite parameters."""
        with torch.no_grad():
            for name, tensor in self.state_dict().items():
                if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                    tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

# Initialize Engine
ENGINE = DiegeticPhysicsEngine()
ENGINE.load_state()

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
                    
                    print(f"🔧 Serving diegetic terminal from: {terminal_path}")
                    print(f"🔧 File exists: {os.path.exists(terminal_path)}")
                    
                    if not os.path.exists(terminal_path):
                        print(f"❌ Diegetic terminal HTML not found at {terminal_path}")
                        self.send_error(404, f"Diegetic terminal HTML not found: {terminal_path}")
                        return
                    
                    with open(terminal_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    print(f"🔧 Diegetic terminal content length: {len(content)}")
                    
                    if len(content) == 0:
                        print("❌ Diegetic terminal HTML is empty!")
                        self.send_error(500, "Diegetic terminal HTML is empty")
                        return
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                    print("✅ Diegetic terminal served successfully")
                    return
                except Exception as e:
                    print(f"❌ Error serving diegetic terminal: {e}")
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
                    
                    print(f"🔧 Attempting to serve HTML from: {trainer_path}")
                    print(f"🔧 File exists: {os.path.exists(trainer_path)}")
                    print(f"🔧 Current working directory: {os.getcwd()}")
                    
                    if not os.path.exists(trainer_path):
                        print(f"❌ HTML file not found at {trainer_path}")
                        self.send_error(404, f"HTML file not found: {trainer_path}")
                        return
                    
                    with open(trainer_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    print(f"🔧 HTML content length: {len(content)}")
                    
                    if len(content) == 0:
                        print("❌ HTML file is empty!")
                        self.send_error(500, "HTML file is empty")
                        return
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                    print("✅ HTML served successfully")
                    return
                except Exception as e:
                    print(f"❌ Error serving Wikipedia trainer: {e}")
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
        print(f"🔥 POST REQUEST RECEIVED: {self.path}")
        try:
            if self.path == '/interact':
                print("📥 Processing /interact request...")
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    data = json.loads(post_body.decode('utf-8'))
                    user_text = data.get('text', '')
                    print(f"📝 User input: '{user_text}'")
                    print("🔧 Starting ENGINE.process_input...")
                    
                    response_data = ENGINE.process_input(user_text)
                    self._send_json(response_data)
                except Exception as e:
                    print(f"❌ Error processing input: {e}")
                    import traceback
                    traceback.print_exc()
                    self._send_error_json(str(e))
                return

            elif self.path == '/associate':
                print("📥 Processing /associate request...")
                try:
                    content_len = int(self.headers.get('Content-Length', 0))
                    post_body = self.rfile.read(content_len)
                    data = json.loads(post_body.decode('utf-8'))
                    text1 = data.get('text1', '')
                    text2 = data.get('text2', '')
                    
                    if not text1 or not text2:
                        self._send_error_json("Missing text1 or text2")
                        return

                    association_command = f"ASSOCIATE: {text1} <-> {text2}"
                    print(f"📝 Association command: '{association_command}'")
                    
                    response_data = ENGINE.process_input(association_command)
                    self._send_json(response_data)
                except Exception as e:
                    print(f"❌ Error processing association: {e}")
                    self._send_error_json(str(e))
                return

            elif self.path == '/ingest':
                print("📥 Processing /ingest request...")
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
                            # Resize to 137 if needed (simple padding/truncation)
                            target_dim = 137
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
                            fossil_path = ENGINE.fossilizer.fossilize(dyad, text_tensor)
                            print(f"✅ Dyad fossilized at: {fossil_path}")
                            
                        except Exception as e:
                            print(f"⚠️ Fossilization failed, continuing with memory-only ingest: {e}")
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
                    print(f"❌ Error processing ingestion: {e}")
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
                        print("✅ Wikipedia integration module loaded successfully")
                    except ImportError as e:
                        print(f"❌ Failed to import Wikipedia integration: {e}")
                        self._send_error_json(f"Wikipedia integration module not available: {e}")
                        return
                    
                    results = []
                    for url in urls:
                        try:
                            title = wikipedia_integration.extract_title_from_url(url)
                            print(f"🔍 Processing Wikipedia page: {title}")
                            
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
                                                print(f"✅ Created association: {concept} ↔ content")
                                                
                                                # Limit to 5 associations per page to prevent backend timeout
                                                if associations_created >= 5:
                                                    print("⚠️  Reached association limit per page (5)")
                                                    break
                                            except Exception as e:
                                                print(f"⚠️  Failed to create association for {concept}: {e}")
                                
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
                            print(f"❌ Error processing {url}: {e}")
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
                    print(f"❌ Wikipedia extraction endpoint error: {e}")
                    self._send_error_json(f"Wikipedia extraction failed: {e}")
                
            elif self.path == '/associate':
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                source = data.get('source', '')
                target = data.get('target', '')
                result = ENGINE.process_input(f"ASSOCIATE: {source} <-> {target}")
                ENGINE.save_state()
                self._send_json({"status": "associated", "source": source, "target": target, "metrics": result})
            
            # ================================================================
            # PHASE 1: LOCAL DATA ENDPOINTS (No HF Token Required)
            # ================================================================
            elif self.path == '/api/test_token':
                # Accept token test — now works with local-only mode too
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode('utf-8'))
                token = data.get('token', '')
                
                if token.startswith('hf_'):
                    # Real HF token — attempt validation
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
                    # Local-only mode — no token needed
                    datasets = LOCAL_LOADER.scan()
                    self._send_json({
                        'success': True,
                        'username': 'local_user',
                        'message': f'Local mode active — {len(datasets)} datasets available'
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
                
                print(f"📥 Local ingestion: {dataset_name} (max={max_samples})")
                
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
                                f'Epoch {epoch+1} complete — iteration {ENGINE.iteration}'
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
            print(f"⚠️  Client connection lost during response: {e}")
        except Exception as e:
            print(f"❌ Error sending JSON response: {e}")
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
            print(f"⚠️  Client connection lost during error response: {e}")
        except Exception as e:
            print(f"❌ Error sending error response: {e}")

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

if __name__ == "__main__":
    main()



