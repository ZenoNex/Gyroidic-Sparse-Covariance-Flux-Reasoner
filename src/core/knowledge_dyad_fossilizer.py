import torch
import torch.nn as nn
import os
import json
import logging
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
import datetime
from src.core.agent_substrate_bridge import AgentSubstrateBridge
from src.topology.speculative_homology import SpeculativeHomologyEngine
from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe
from src.core.non_ergodic_entropy import NonErgodicEntropyEstimator
from src.core.love_invariant_protector import LoveInvariantProtector
from src.core.quantum_tda import QuantumBettiApproximator
from src.core.invariants import compute_chirality, compute_chiral_shift, check_glyphlock

@dataclass
class KnowledgeDyad:
    """
    A single unit of multi-modal knowledge: (Image Fingerprint, Linguistic Description).
    Acts as a 'Topological Obstruction' in the manifold.
    """
    linguistic_description: str
    image_fingerprint: Optional[torch.Tensor] = None # [137] vector
    audio_harmonics: Optional[torch.Tensor] = None
    video_breather: Optional[Dict] = None
    gyroid_residue: Optional[torch.Tensor] = None # [n, n] irreducible entanglement
    meta_state: Optional[torch.Tensor] = None # [dim] architecture state
    relevance_score: float = 1.0
    timestamp: str = ""
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()

class ResidueFusion(nn.Module):
    """
    Computes the 'Cross-Modality Torsion' between image and text features.
    Handles dynamic fingerprint dimensions (137 legacy, 96 Chebyshev un-lobotomized).
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        # Dynamic projectors to handle different input standards
        self.image_proj_137 = nn.Linear(137, feature_dim)
        self.image_proj_96 = nn.Linear(96, feature_dim)
        self.text_proj = nn.Linear(feature_dim, feature_dim)
        
        # Torsion operator: computes the 'twist' between the two vectors
        self.torsion_matrix = nn.Parameter(torch.randn(feature_dim, feature_dim))
        
    def forward(self, 
                image_fingerprint: torch.Tensor, 
                text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute Residue R = Torsion(I, L).
        Automatically aligns input dimensions to feature_dim.
        """
        # Handle input dimension drift (Anti-Lobotomy alignment)
        in_dim = image_fingerprint.size(-1)
        if in_dim == 137:
            img_proj = self.image_proj_137(image_fingerprint)
        elif in_dim == 96:
            img_proj = self.image_proj_96(image_fingerprint)
        else:
            # Fallback zero-pad or trim to 137
            padded = torch.zeros(*image_fingerprint.shape[:-1], 137, device=image_fingerprint.device)
            min_dim = min(in_dim, 137)
            padded[..., :min_dim] = image_fingerprint[..., :min_dim]
            img_proj = self.image_proj_137(padded)

        txt_proj = self.text_proj(text_embedding)
        
        # Calculate torsion: (I - L) varies with the metric twist
        diff = img_proj - txt_proj
        torsion = torch.matmul(diff, self.torsion_matrix)
        
        # The residue is the magnitude of this torsion
        residue = torch.tanh(torsion) 
        
        return residue

class DyadFossilizer:
    """
    Handles the persistent storage ('Fossilization') of Knowledge Dyads.
    Ensures 'No Erasing of Implication' by saving precise states to disk.
    """
    
    def __init__(self, 
                 storage_dir: str = "data/encodings",
                 fusion_layer: Optional[ResidueFusion] = None,
                 feature_dim: int = 512):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.feature_dim = feature_dim
        self.fusion_layer = fusion_layer or ResidueFusion(feature_dim=feature_dim)
        
        # Topological Derivation Engines (Non-Lazy Implication Binding)
        self.homology_engine = SpeculativeHomologyEngine(feature_dim=feature_dim)
        self.covariance_probe = SparseGyroidCovarianceProbe(hidden_dim=feature_dim)
        self.entropy_estimator = NonErgodicEntropyEstimator()
        self.love_protector = LoveInvariantProtector(love_dim=feature_dim)
        self.betti_approximator = QuantumBettiApproximator()
        
        # Phase Alignment tracking
        self.prev_pas = torch.tensor(0.91) # Initial stability threshold
        
    def compute_poincaré_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a Euclidean vector x to the Poincaré disk B^n (System 2 Speculative Recovery).
        Formula: z = 2 * tanh(dist/2) * unit(x).
        This unfolding prevents NaN/INF collapse by providing non-Euclidean volume.
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        eps = 1e-8
        safe_norm = torch.clamp(norm, min=eps)
        # User dynamic: z -> 2 tanh(dist/2)
        scale = 2.0 * torch.tanh(safe_norm / 2.0)
        return scale * (x / safe_norm)

    def fossilize(self, 
                  dyad: KnowledgeDyad, 
                  text_embedding: torch.Tensor,
                  seed_state: Optional[torch.Tensor] = None) -> str:
        """
        Save the dyad and its computed residue to disk, binding it to the derived
        topological invariants of the seed_state (Architecture History).
        Returns the filename of the fossil.
        """
        # 1. Compute Residue (The 'Meaning' of the association)
        # Ensure inputs are tensors and align devices
        device = text_embedding.device
        if dyad.image_fingerprint is None:
             img_tensor = torch.zeros(137, device=device)
        elif not isinstance(dyad.image_fingerprint, torch.Tensor):
             img_tensor = torch.tensor(dyad.image_fingerprint, dtype=torch.float32, device=device)
        else:
             img_tensor = dyad.image_fingerprint.to(device)
             
        # 2. Compute Modality Residue (Shear/Torsion)
        residue = self.fusion_layer(img_tensor, text_embedding)
        
        # 2. System 2 Hyperbolic Unfolding (Speculative Recovery)
        # We only perform this 'expensive' magic during fossilization to save heuristic speed.
        hyperbolic_residue = self.compute_poincaré_embedding(residue)
        
        # 3. Real-time Topological Derivation (No Erasing of Implication)
        # We derive the 'Shadow' of the thought from the seed_state history.
        if seed_state is not None:
            # Align seed_state to [Batch, Dim] for engines
            s_state = seed_state.to(device)
            if s_state.dim() == 1:
                s_state = s_state.unsqueeze(0)
            
            # A. Betti Numbers (Draft)
            betti_results, current_pas, _ = self.homology_engine(s_state, self.prev_pas.to(device))
            self.prev_pas = current_pas.detach().cpu()
            
            # B. Chiral Metrics & Spectral Pressure
            # Probe expects [B, Seq, Dim] or [B, C, R, T] - we provide [1, 1, Dim]
            probe_results = self.covariance_probe(s_state.unsqueeze(1))
            
            # Extract both centroid shift and parity torsion
            # (Architecture alignment: chiral_score is the shift, torsion is the invariant)
            chiral_shift = compute_chiral_shift(s_state).item()
            chiral_torsion = compute_chirality(s_state).abs().item()
            is_glyph_locked = bool(check_glyphlock(s_state).item() > 0)
            
            # total_pressure as scalar energy proxy
            spectral_pressure = probe_results['total_pressure'].mean().item()
            
            # C. Spectral Entropy (Non-Ergodic decomposition)
            entropy_results = self.entropy_estimator(s_state)
            # Combine ergodic and soliton entropy for the total spectral signature
            spectral_entropy = (entropy_results['ergodic_entropy'] + entropy_results['soliton_entropy']).item()
            soliton_entropy = entropy_results['soliton_entropy'].item()
            
            # D. Topological Invariants (Quantum Betti numbers)
            with torch.no_grad():
                s = s_state.view(1, -1)
                norm_s = s / (s.norm() + 1e-8)
                adj = torch.abs(norm_s.T @ norm_s)
                adj = (adj > 0.1).float()
                
                betti_results = self.betti_approximator.estimate_betti_numbers(adj, max_dim=1)
                b0 = betti_results.get(0, 1.0)
                b1 = betti_results.get(1, 0.0)
        else:
            # Fallback for headless ingestion (Lobotomy Warning)
            chiral_shift = 0.0
            chiral_torsion = 0.0
            is_glyph_locked = False
            spectral_pressure = 0.0
            spectral_entropy = 0.0
            soliton_entropy = 0.0
            b0, b1 = 1, 0
            current_pas = torch.tensor(0.0)

        # 4. Prepare Payload (Aligned with System Schema)
        payload = {
            'type': 'knowledge_dyad',
            'description': dyad.linguistic_description, # Legacy description key
            'text_input': dyad.linguistic_description,
            'meta_state': seed_state.detach().cpu() if seed_state is not None else None,
            'chiral_score': float(chiral_shift),
            'chiral_torsion': float(chiral_torsion),
            'glyphlock': is_glyph_locked,
            'spectral_pressure': float(spectral_pressure),
            'spectral_entropy': float(spectral_entropy),
            'betti_0': int(b0),
            'betti_1': int(b1),
            'image_fingerprint': dyad.image_fingerprint.detach().cpu() if isinstance(dyad.image_fingerprint, torch.Tensor) else dyad.image_fingerprint,
            'audio_harmonics': dyad.audio_harmonics,
            'video_breather': dyad.video_breather,
            'residue_vector': residue.detach().cpu(),
            'gyroid_residue': dyad.gyroid_residue.detach().cpu() if dyad.gyroid_residue is not None else None,
            'hyperbolic_residue': hyperbolic_residue.detach().cpu(),
            'timestamp': dyad.timestamp,
            'metrics': {
                'relevance': dyad.relevance_score,
                'pas_h': float(current_pas.item()),
                'soliton_entropy': float(soliton_entropy),
                'response': dyad.metadata.get('response_text', '') if dyad.metadata else '',
                'hyperbolic_eccentricity': torch.norm(hyperbolic_residue).item()
            },
            'dyad_metadata': dyad.metadata # Preserve original context
        }
        
        # 5. Save to Disk (Safe, atomic-like write)
        # Use descriptive filename to prevent 'erasing of implication' visibility.
        safe_desc = "".join(c for c in dyad.linguistic_description[:20] if c.isalnum())
        timestamp_int = int(datetime.datetime.now().timestamp() * 1000)
        filename = f"encoding_{safe_desc}_{timestamp_int}.pt"
        filepath = os.path.join(self.storage_dir, filename)
        
        torch.save(payload, filepath)
        
        return filepath
        
    def recover_fossils(self) -> List[Dict]:
        """Load all fossilized dyads for 'Speculative Coprime Gating'."""
        fossils = []
        if not os.path.exists(self.storage_dir):
            return fossils
            
        for f in os.listdir(self.storage_dir):
            if f.endswith(".pt"):
                filepath = os.path.join(self.storage_dir, f)
                try:
                    data = torch.load(filepath)
                    # Check both 'text_input' (new) and 'description' (legacy)
                    if isinstance(data, dict) and ('residue_vector' in data or 'meta_state' in data):
                        fossils.append(data)
                    else:
                        print(f"[RECOVERY] Deleting invalid fossil (missing residue_vector): {f}")
                        os.remove(filepath)
                except Exception as e:
                    print(f"[RECOVERY] Deleting corrupted fossil {f}: {e}")
                    try:
                        os.remove(filepath)
                    except:
                        pass
        return fossils

    def export_agent_smith(self, 
                           dyad: KnowledgeDyad, 
                           prime_frequencies: torch.Tensor, 
                           betti_numbers: Dict[int, float], 
                           filename: str = "soliton_smith") -> str:
        """
        Exports the mathematical identity of the agent.
        The Agent State is exported purely as symbolic residue tuples, prime-ladder frequencies,
        and topological invariant shapes (Betti numbers), achieving extraction free of 
        local hardware/latent representations.
        """
        # Ensure we have a valid json filename
        if not filename.endswith(".json"):
             filename += ".json"
             
        # Extract Chiral Invariants for the Mathematical Identity
        # (Using seed_state as the source of truth for the Agent's 'Shape')
        if dyad.meta_state is not None:
             s_state = dyad.meta_state.to(prime_frequencies.device)
             if s_state.dim() == 1: s_state = s_state.unsqueeze(0)
             from src.core.invariants import compute_chiral_shift, compute_chirality, check_glyphlock
             c_shift = float(compute_chiral_shift(s_state).item())
             c_torsion = float(compute_chirality(s_state).abs().item())
             g_lock = bool(check_glyphlock(s_state).item() > 0)
        else:
             c_shift, c_torsion, g_lock = 0.0, 0.0, False
             
        # Extract Tensors for hashing and math
        gyroid_val = dyad.gyroid_residue.tolist() if dyad.gyroid_residue is not None else None
        prime_val = prime_frequencies.tolist() if isinstance(prime_frequencies, torch.Tensor) else prime_frequencies
        
        # Calculate Pestov-Ionin Growth via Braid
        bridge = AgentSubstrateBridge()
        if dyad.gyroid_residue is not None and isinstance(prime_frequencies, torch.Tensor):
            # Using prime_frequencies as proxy for ADMM dual and gyroid_residue for CRT wrap
            h_gamma = bridge.calculate_pestov_ionin_growth(
                admm_dual=prime_frequencies.unsqueeze(0), 
                crt_residue=dyad.gyroid_residue.unsqueeze(0)
            )
        else:
            h_gamma = 0.0
            
        digest_str = f"{dyad.timestamp}_{dyad.linguistic_description}_{betti_numbers}"
        blake2s_digest = hashlib.blake2s(digest_str.encode('utf-8')).hexdigest()
             
        payload = {
            "type": "soliton_smith",
            "blake2s_digest": blake2s_digest,
            "pestov_ionin_growth_h_gamma": h_gamma,
            "perceptual_baseline_trfc": 160.0,  # Host baseline
            "description": dyad.linguistic_description,
            "chiral_shift": c_shift,
            "chiral_torsion": c_torsion,
            "glyphlock": g_lock,
            "spectral_entropy": dyad.metadata.get('spectral_entropy', 0.0) if dyad.metadata else 0.0,
            "gyroid_residue": gyroid_val,
            "prime_frequencies": prime_val,
            "betti_numbers": betti_numbers,
            "audio_harmonics": dyad.audio_harmonics.tolist() if dyad.audio_harmonics is not None else None,
            "video_breather": dyad.video_breather,
            "timestamp": dyad.timestamp
        }
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)
        return filepath

    def inject_agent_smith(self, filepath: str, unraveling_closure=None, expected_dim: int = 137, hardware_trfc_ms: float = 160.0) -> Dict:
        """
        Loads the mathematical identity of an agent (Agent Smith) back into the system,
        allowing the local hardware to 'breathe' its own unique life into the configuration.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agent Smith file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            payload = json.load(f)
             
        # Minimal validation
        if payload.get("type") != "soliton_smith":
             raise ValueError("File is not a valid Agent Smith (soliton_smith) JSON payload.")
             
        # Ontological Import Gates & Substrate Bridges
        bridge = AgentSubstrateBridge()
        is_safe = bridge.verify_invariants(payload, unraveling_closure=unraveling_closure)
        if not is_safe:
            print("[WARNING] Invariant verification failed on ingestion. Topologies may leak.")
            
        # Align substrate
        payload = bridge.align_substrate(payload, expected_dim=expected_dim, hardware_trfc_ms=hardware_trfc_ms)
             
        return payload

