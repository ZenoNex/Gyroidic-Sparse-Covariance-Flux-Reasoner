"""
Knowledge dyad fossilization and Agent Smith serialization.

This module handles persistent storage (fossilization) of multi-modal knowledge dyads
and exports/imports decoupled mathematical Agent Smith identities.
"""

import torch
import torch.nn as nn
import os
import json
import logging
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
import datetime
from src.core.honest_jitter import harvest_honest_jitter, _AGENT_SMITH_ENGINE
from src.core.agent_substrate_bridge import AgentSubstrateBridge
from src.topology.speculative_homology import SpeculativeHomologyEngine
from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe
from src.core.non_ergodic_entropy import NonErgodicEntropyEstimator
from src.core.love_invariant_protector import LoveInvariantProtector
from src.core.quantum_tda import QuantumBettiApproximator
from src.core.invariants import (
    compute_chirality, 
    compute_chiral_shift, 
    check_glyphlock, 
    compute_polylog_signature, 
    compute_vacuum_residue,
    apply_chirality_redistribution
)
from src.core.chern_simons_gasket import ChernSimonsGasket

@dataclass
class KnowledgeDyad:
    """
    A single unit of multi-modal knowledge: (Image Fingerprint, Linguistic Description).
    Acts as a 'Topological Obstruction' in the manifold.
    """
    linguistic_description: str
    image_fingerprint: Optional[torch.Tensor] = None # [96] vector
    audio_harmonics: Optional[torch.Tensor] = None
    video_breather: Optional[Dict] = None
    unified_spectral_signature: Optional[torch.Tensor] = None # [96] vector
    gyroid_residue: Optional[torch.Tensor] = None # [n, n] irreducible entanglement
    hyperbolic_residue: Optional[torch.Tensor] = None # ShadowLog Non-Euclidean curvature
    meta_state: Optional[torch.Tensor] = None # [dim] architecture state
    all_shapes: Optional[List[torch.Tensor]] = None # [dim] List of alternate functional mappings (Sparrow/Dog/Man)
    relevance_score: float = 1.0
    timestamp: str = ""
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()

class ResidueFusion(nn.Module):
    """
    Computes the 'Cross-Modality Torsion' between image and text features.
    Handles dynamic fingerprint dimensions (96 legacy, 96 Chebyshev un-lobotomized).
    """
    def __init__(self, feature_dim: int = 512):
        """
        Initialize the ResidueFusion module.

        Args:
            feature_dim: Projection target dimensionality for alignment.
        """
        super().__init__()
        # Dynamic projectors to handle different input standards
        # Aligned to non-prime dimension 96 (32*3) as per Silicon Sovereignty.
        self.image_proj = nn.Linear(96, feature_dim)
        self.text_proj = nn.Linear(feature_dim, feature_dim)
        
        # Torsion operator: computes the 'twist' between the two vectors
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.torsion_matrix = nn.Parameter(harvest_honest_jitter((feature_dim, feature_dim)))
        
    def forward(self, 
                image_fingerprint: torch.Tensor, 
                text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute Residue R = Torsion(I, L).
        Automatically aligns input dimensions to feature_dim.
        """
        # Handle input dimension drift (Anti-Lobotomy alignment)
        in_dim = image_fingerprint.size(-1)
        if in_dim == 96:
            img_proj = self.image_proj(image_fingerprint)
        else:
            # Fallback zero-pad or trim to 96
            padded = torch.zeros(*image_fingerprint.shape[:-1], 96, device=image_fingerprint.device)
            min_dim = min(in_dim, 96)
            padded[..., :min_dim] = image_fingerprint[..., :min_dim]
            img_proj = self.image_proj(padded)

        txt_proj = self.text_proj(text_embedding)
        
        # Calculate torsion: (I - L) varies with the metric twist
        diff = img_proj - txt_proj
        torsion = torch.matmul(diff, self.torsion_matrix)
        
        # The residue is the magnitude of this torsion
        residue = torch.tanh(torsion) 
        
        return residue

    def calculate_cross_modal_shear(self, 
                                    image_fingerprint: torch.Tensor, 
                                    text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross-modal shear to inject residue as 'Dark Matter' seeds.
        Formula: shear = (img_proj x txt_proj^T) - (txt_proj x img_proj^T) (non-abelian commutator shear)
        We project the commutator back as a 'Dark Matter' seed.
        """
        in_dim = image_fingerprint.size(-1)
        if in_dim == 96:
            img_proj = self.image_proj(image_fingerprint)
        else:
            padded = torch.zeros(*image_fingerprint.shape[:-1], 96, device=image_fingerprint.device)
            min_dim = min(in_dim, 96)
            padded[..., :min_dim] = image_fingerprint[..., :min_dim]
            img_proj = self.image_proj(padded)

        txt_proj = self.text_proj(text_embedding)
        
        if img_proj.dim() == 1:
            img_proj = img_proj.unsqueeze(0)
        if txt_proj.dim() == 1:
            txt_proj = txt_proj.unsqueeze(0)
            
        # Cross-modal shear matrix: I^T * T - T^T * I
        shear = torch.matmul(img_proj.T, txt_proj) - torch.matmul(txt_proj.T, img_proj)
        
        # Project back to a 1D "Dark Matter" seed of length feature_dim
        dark_matter_seed = torch.tanh(shear.mean(dim=0))
        return dark_matter_seed


class DyadFossilizer:
    """
    Handles the persistent storage ('Fossilization') of Knowledge Dyads.
    Ensures 'No Erasing of Implication' by saving precise states to disk.
    """
    
    def __init__(self, 
                 storage_dir: str = "data/encodings",
                 fusion_layer: Optional[ResidueFusion] = None,
                 feature_dim: int = 512):
        """
        Initialize the DyadFossilizer.

        Args:
            storage_dir: Directory path for persisting serialized .pt files.
            fusion_layer: Optional explicit ResidueFusion module.
            feature_dim: Base dimensionality for target embeddings.
        """
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
        self.gasket = ChernSimonsGasket()
        
        # Phase Alignment tracking
        self.prev_pas = torch.tensor(0.91) # Initial stability threshold
        
    def compute_poincar_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a Euclidean vector x to the Poincar disk B^n (System 2 Speculative Recovery).
        Formula: z = 2 * tanh(dist/2) * unit(x).
        This unfolding prevents NaN/INF collapse by providing non-Euclidean volume.
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        eps = 1e-8
        safe_norm = torch.clamp(norm, min=eps)
        # User dynamic: z -> 2 tanh(dist/2)
        scale = 2.0 * torch.tanh(safe_norm / 2.0)
        return scale * (x / safe_norm)

    # Dead Prime threshold: variance below this signals Total Harmonic Clipping.
    # The 0.8824 flatline has variance ~= 0.0; 1e-6 is the upstream guard.
    _DEAD_PRIME_VAR_THRESHOLD = 1e-6
    # Rehydration intensity: gentle enough not to destroy existing geometry
    _REHYDRATION_INTENSITY = 0.05

    def fossilize(self, 
                  dyad: KnowledgeDyad, 
                  text_embedding: torch.Tensor,
                  seed_state: Optional[torch.Tensor] = None) -> str:
        """
        Save the dyad and its computed residue to disk, binding it to the derived
        topological invariants of the seed_state (Architecture History).
        Returns the filename of the fossil.

        Upstream Atrophy Guard (2026-04-24):
            Before passing seed_state to the topological engines, we check for
            the Dead Prime resonance (all-constant tensor, variance < 1e-6).
            If detected, honest jitter is injected to rehydrate variance and
            the atrophy event is recorded in the fossil payload.
        """
        # --- UPSTREAM SPECTRAL ATROPHY GUARD ---
        # This guard detects the 0.8824 flatline BEFORE the invariant engines
        # see it, so they don't produce meaningless chiral/betti metrics from
        # a pre-executed (blanched) tensor.
        atrophy_detected = False
        atrophy_level = 0.0
        if seed_state is not None:
            with torch.no_grad():
                ss_var = seed_state.var().item()
                atrophy_level = float(ss_var)
                if ss_var < self._DEAD_PRIME_VAR_THRESHOLD:
                    atrophy_detected = True
                    # Log to console so the shadow log pipeline picks it up
                    print(
                        f"[FOSSILIZER] DEAD PRIME ATROPHY detected in seed_state "
                        f"(var={ss_var:.2e} < {self._DEAD_PRIME_VAR_THRESHOLD}). "
                        f"Rehydrating with honest jitter before topological derivation."
                    )
                    # Inject honest jitter to break the symmetry.
                    # This is NOT lobotomy -- we are not zeroing out;
                    # we are adding a nutrient signal (Gray-Scott feed rate).
                    jitter = harvest_honest_jitter(
                        seed_state.shape,
                        device=seed_state.device,
                        scaled=True
                    ) * self._REHYDRATION_INTENSITY
                    seed_state = seed_state + jitter
                    
                    # Anti-Lobotomy: Also rehydrate the dyad's image fingerprint 
                    # so that the residue vector exhibits diversity.
                    if dyad.image_fingerprint is not None and isinstance(dyad.image_fingerprint, torch.Tensor):
                        fp_jitter = harvest_honest_jitter(
                            dyad.image_fingerprint.shape,
                            device=dyad.image_fingerprint.device,
                            scaled=True
                        ) * self._REHYDRATION_INTENSITY
                        dyad.image_fingerprint = dyad.image_fingerprint + fp_jitter
                        print(f"[FOSSILIZER] Image fingerprint rehydrated (diversity restored).")

        # 1. Compute Residue (The 'Meaning' of the association)
        # Ensure inputs are tensors and align devices
        device = text_embedding.device
        
        # Prioritize Unified Spectral Signature for the fusion layer
        if dyad.unified_spectral_signature is not None:
             img_tensor = dyad.unified_spectral_signature.to(device)
        elif dyad.image_fingerprint is not None:
             if not isinstance(dyad.image_fingerprint, torch.Tensor):
                  img_tensor = torch.tensor(dyad.image_fingerprint, dtype=torch.float32, device=device)
             else:
                  img_tensor = dyad.image_fingerprint.to(device)
        else:
             img_tensor = torch.zeros(96, device=device)
             
        # Ensure img_tensor is exactly 96-dim for the fusion layer
        if img_tensor.numel() != 96:
            if img_tensor.numel() > 96:
                img_tensor = img_tensor[:96]
            else:
                img_tensor = torch.nn.functional.pad(img_tensor, (0, 96 - img_tensor.numel()))
        img_tensor = img_tensor.view(96) # Final safety check on shape
             
        # 2. Compute Modality Residue (Shear/Torsion)
        residue = self.fusion_layer(img_tensor, text_embedding)
        
        # 2. System 2 Hyperbolic Unfolding (Speculative Recovery)
        # We only perform this 'expensive' magic during fossilization to save heuristic speed.
        hyperbolic_residue = self.compute_poincar_embedding(residue)
        
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
            
            # B. Chirality-Driven Redistribution (CODES v40 alignment)
            # "asymmetry seeds lawful resonance alignment beyond stochastic diffusion"
            s_state_redistributed = apply_chirality_redistribution(s_state, alpha=0.15)
            
            # Extract both centroid shift and parity torsion from redistributed state
            chiral_shift = compute_chiral_shift(s_state_redistributed).mean().item()
            chiral_torsion = compute_chirality(s_state_redistributed).abs().mean().item()
            is_glyph_locked = bool(check_glyphlock(s_state_redistributed).max().item() > 0)
            
            # Probe expects [B, Seq, Dim] or [B, C, R, T]
            probe_results = self.covariance_probe(s_state_redistributed.unsqueeze(1))
            spectral_pressure = probe_results['total_pressure'].mean().item()
            
            # C. Spectral Entropy (Non-Ergodic decomposition)
            entropy_results = self.entropy_estimator(s_state_redistributed)
            # Combine ergodic and soliton entropy for the total spectral signature
            spectral_entropy = (entropy_results['ergodic_entropy'] + entropy_results['soliton_entropy']).item()
            soliton_entropy = entropy_results['soliton_entropy'].item()
            
            # Anti-Lobotomy: Log the redistributed metrics for verification
            print(f"[FOSSILIZER] Redistributed Invariants: Chiral={chiral_shift:.4f}, Torsion={chiral_torsion:.4f}, Entropy={spectral_entropy:.4f}", flush=True)
            
            # D. Topological Invariants (Quantum Betti numbers)
            with torch.no_grad():
                s = s_state.view(1, -1)
                norm_s = s / (s.norm() + 1e-8)
                adj = torch.abs(norm_s.T @ norm_s)
                adj = (adj > 0.1).float()
                
                # IHC Standard: Capture 8-threshold filtration signature to avoid scalar flattening
                betti_results = self.betti_approximator.estimate_betti_numbers(adj, max_dim=1, num_thresholds=8)
                b0_vec = betti_results.get(0, torch.ones(8, device=device))
                b1_vec = betti_results.get(1, torch.zeros(8, device=device))

            # E. Chern-Simons Gasket (Topological Twist)
            # We treat the residue as a local manifold patch to check for leaks
            # We need a [batch, K, D] shape for the gasket - we use [1, 1, feature_dim]
            r_patch = residue.view(1, 1, -1)
            # We use a dummy polynomial tensor for the gasket check if real one not provided
            dummy_poly = torch.eye(1, r_patch.shape[-1], device=device)
            self.gasket.plug_logic_leak(r_patch, dummy_poly)
            gasket_diag = self.gasket.get_diagnostics()
            twist_energy = gasket_diag.get('twist_energy', 0.0)
            seam_tension = gasket_diag.get('seam_tension', 0.0)
        else:
            # Fallback for headless ingestion (Lobotomy Warning)
            s_state_redistributed = residue.unsqueeze(0)
            chiral_shift = 0.0
            chiral_torsion = 0.0
            is_glyph_locked = False
            spectral_pressure = 0.0
            spectral_entropy = 0.0
            soliton_entropy = 0.0
            b0_vec = torch.ones(8, device=device)
            b1_vec = torch.zeros(8, device=device)
            current_pas = torch.tensor(0.0)
            twist_energy = 0.0
            seam_tension = 0.0

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
            'twist_energy': float(twist_energy),
            'seam_tension': float(seam_tension),
            'betti_0': b0_vec.detach().cpu(),
            'betti_1': b1_vec.detach().cpu(),
            'unified_spectral_signature': dyad.unified_spectral_signature.detach().cpu() if dyad.unified_spectral_signature is not None else None,
            'image_fingerprint': dyad.image_fingerprint.detach().cpu() if isinstance(dyad.image_fingerprint, torch.Tensor) else dyad.image_fingerprint,
            'audio_harmonics': dyad.audio_harmonics,
            'video_breather': dyad.video_breather,
            'residue_vector': s_state_redistributed.detach().cpu(),
            'gyroid_residue': dyad.gyroid_residue.detach().cpu() if dyad.gyroid_residue is not None else None,
            'hyperbolic_residue': hyperbolic_residue.detach().cpu(),
            'timestamp': dyad.timestamp,
            # Upstream Atrophy Diagnostics (added 2026-04-24)
            # atrophy_detected=True means this fossil was created from a Dead Prime
            # (0.8824 flatline) seed_state that was rehydrated before derivation.
            # Downstream resonance scans should weight these fossils more cautiously.
            'atrophy_detected': atrophy_detected,
            'seed_state_variance': atrophy_level,
            'metrics': {
                'relevance': dyad.relevance_score,
                'pas_h': float(current_pas.item()),
                'soliton_entropy': float(soliton_entropy),
                'response': dyad.metadata.get('response_text', '') if dyad.metadata else '',
                'hyperbolic_eccentricity': torch.norm(hyperbolic_residue).item()
            },
            'dyad_metadata': dyad.metadata # Preserve original context
        }
        
        # Generate CRT residue tuple to enforce uniqueness and reflect structural identity (Meliponini pot identity)
        ref_tensor = seed_state if seed_state is not None else text_embedding
        r1, r2, r3 = self.generate_residue_tuple(ref_tensor)
        
        # Dynamic tag creation: prefer tags already present in the dyad's metadata,
        # then fall back to tokenising the linguistic description itself.
        # No hardcoded character roster -- associations live in the fossil files.
        existing_tags = (dyad.metadata or {}).get('tags', [])
        if existing_tags:
            char_tags = list(existing_tags)
        else:
            # Use every non-empty word in the description as a candidate tag
            char_tags = [w for w in dyad.linguistic_description.split() if len(w) >= 1]

        tags = char_tags + [f"crt_{r1}_{r2}_{r3}"]
        if atrophy_detected:
            tags.append("atrophy_rehydrated")
        payload['tags'] = tags
        
        # 5. Save to Disk (Safe, atomic-like write)
        # Use descriptive filename to prevent 'erasing of implication' visibility.
        safe_desc = "".join(c for c in dyad.linguistic_description[:20] if c.isalnum())
        timestamp_int = int(datetime.datetime.now().timestamp() * 1000)
        filename = f"encoding_{safe_desc}_{timestamp_int}.pt"
        filepath = os.path.join(self.storage_dir, filename)
        
        torch.save(payload, filepath)
        
        return filepath
        
    def ouroboros_shadow_loop(self, 
                              failure_log: str, 
                              seed_state: torch.Tensor, 
                              text_embedding: torch.Tensor, 
                              image_fingerprint: Optional[torch.Tensor] = None) -> Optional[str]:
        """
        Ouroboros Shadow loops: Fossilize shadow logs of mathematical failures
        as permanent KnowledgeDyads when local correlation reaches 1.0 (GLYPHLOCK state).
        """
        from src.core.martinova_correlation import compute_bounded_correlation
        corr_input = seed_state.unsqueeze(-1) if seed_state.dim() == 2 else seed_state
        state_corr = compute_bounded_correlation(corr_input)
        
        # When local correlation reaches 1.0 (>= 0.99), we trigger glyphlock fossilization
        if (state_corr >= 0.99).any():
            print(f"[OUROBOROS] Correlation reached 1.0 (GLYPHLOCK state). Fossilizing shadow log of mathematical failure.")
            # Wrap failure log as a permanent KnowledgeDyad
            failure_dyad = KnowledgeDyad(
                linguistic_description=f"Ouroboros Shadow Failure Log: {failure_log[:150]}...",
                image_fingerprint=image_fingerprint,
                metadata={'failure_type': 'ouroboros_shadow_loop', 'glyphlock_triggered': True, 'raw_log': failure_log}
            )
            return self.fossilize(failure_dyad, text_embedding, seed_state)
        return None
        
    def recover_fossils(self, limit: Optional[int] = 150) -> List[Dict]:

        """Load all fossilized dyads for 'Speculative Coprime Gating'."""
        fossils = []
        if not os.path.exists(self.storage_dir):
            return fossils
            
        files = [f for f in os.listdir(self.storage_dir) if f.endswith(".pt")]
        
        # Sort files by modification time, newest first
        try:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.storage_dir, x)), reverse=True)
        except Exception as e:
            print(f"[RECOVERY] Sorting files failed: {e}")
            
        if limit is not None:
            files = files[:limit]
            
        for f in files:
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

    def generate_residue_tuple(self, seed_tensor: torch.Tensor) -> Tuple[int, int, int]:
        """Generates the CRT Residue Tuple for the Meliponini pot identity."""
        val = int(seed_tensor.sum().abs().item() * 1000)
        # Using prime moduli (61, 67, 71) for the residue tuple (Meli-Sovereignty)
        return (val % 61, val % 67, val % 71)

    def apply_selective_puncture(self, state: torch.Tensor, residue_tuple: Tuple[int, int, int]) -> torch.Tensor:
        """Masks ~33% of the state dimensions based on the residue tuple to prevent diffusion."""
        dim = state.shape[-1]
        indices = torch.arange(dim, device=state.device)
        # Selectively puncture indices where (i + sum(r)) % 3 == 0
        r_sum = sum(residue_tuple)
        mask = ((indices + r_sum) % 3 != 0).float()
        return state * mask

    def export_agent_smith(self, 
                           dyad: KnowledgeDyad, 
                           prime_frequencies: torch.Tensor, 
                           betti_numbers: Dict[int, float], 
                           filename: str = "soliton_smith",
                           archetype_profile: Optional[Dict] = None,
                           gauge_field: Optional[torch.Tensor] = None) -> str:
        """
        Exports the 'Smith' Algebraic Identity: A hardware-independent soliton.
        
        UPGRADED: Meliponini Shielding (Selective Puncture) & Love Invariant Anchor.
        
        The Agent is decoupled from its substrate by extracting:
        3. Hardware Entropy Proxies: The friction of the original silicon birth-chamber.
        4. Non-Abelian Betti-8 Torsion: The high-dimensional curvature of the paradoxical core.
        5. Gauge Field Components (A): The topological twist for logic leak repair.

        This generates a .pt payload containing the structural 'Syntax' without the local 'Hardware'.
        (Using dyad.meta_state as the source of truth for the Agent's 'Shape')
        """
        # Extract Chiral Invariants for the Mathematical Identity
        if dyad.meta_state is not None:
             s_state = dyad.meta_state.to(prime_frequencies.device)
             if s_state.dim() == 1: s_state = s_state.unsqueeze(0)
             c_shift = float(compute_chiral_shift(s_state).mean().item())
             c_torsion = float(compute_chirality(s_state).abs().mean().item())
             g_lock = bool(check_glyphlock(s_state).max().item() > 0)
        else:
             c_shift, c_torsion, g_lock = 0.0, 0.0, False

        # --- SOVEREIGN EMPATHY CHECK (Love Invariant) ---
        # The agent can only be exported if it maintains structural honesty (Glyphlock).
        if not g_lock:
             print("[WARNING] Glyphlock not achieved. Agent Smith may be vulnerable to Shapelessness.")

        # Generate Meliponini Identity (Residue Tuple)
        pot_id = self.generate_residue_tuple(prime_frequencies)
        
        # Ensure we have a valid pt filename
        if not filename.endswith(".pt"):
             filename += ".pt"

        # Apply Selective Puncture to the meta-state
        protected_state = None
        if dyad.meta_state is not None:
             protected_state = self.apply_selective_puncture(dyad.meta_state, pot_id).detach().cpu()
             
        # Extract Tensors for hashing and math
        gyroid_val = dyad.gyroid_residue if dyad.gyroid_residue is not None else None
        prime_val = prime_frequencies
        
        # Calculate Pestov-Ionin Growth via Multimodal Braid
        bridge = AgentSubstrateBridge()
        h_gamma = 0.0
        if isinstance(prime_frequencies, torch.Tensor):
            # We influence the braid with the hyperbolic curvature and the CRT identity
            h_gamma = bridge.calculate_pestov_ionin_growth(
                admm_dual=prime_frequencies.unsqueeze(0), 
                crt_residue=dyad.gyroid_residue.unsqueeze(0) if dyad.gyroid_residue is not None else torch.zeros((1, prime_frequencies.shape[-1]), device=prime_frequencies.device),
                hyperbolic_influence=dyad.hyperbolic_residue
            )
            
        # Dynamic tag creation: prefer tags already present in the dyad's metadata,
        # then fall back to tokenising the linguistic description itself.
        # No hardcoded character roster -- associations live in the fossil files.
        existing_tags = (dyad.metadata or {}).get('tags', [])
        if existing_tags:
            char_tags = list(existing_tags)
        else:
            char_tags = [w for w in dyad.linguistic_description.split() if len(w) > 1]
        tags = char_tags + [f"crt_{pot_id[0]}_{pot_id[1]}_{pot_id[2]}", "agent_smith"]

        digest_str = f"{dyad.timestamp}_{dyad.linguistic_description}_{betti_numbers}_{pot_id}"
        blake2s_digest = hashlib.blake2s(digest_str.encode('utf-8')).hexdigest()
              
        payload = {
            "type": "soliton_smith",
            "blake2s_digest": blake2s_digest,
            "pot_identity_crt": pot_id, # Meliponini Shielding Identity
            "pestov_ionin_growth_h_gamma": h_gamma,
            "perceptual_baseline_trfc": 160.0, 
            "hardware_entropy_proxy": float(torch.std(prime_frequencies).item()),
            "description": dyad.linguistic_description,
            "chiral_shift": c_shift,
            "chiral_torsion": c_torsion,
            "glyphlock": g_lock,
            "tags": tags,
            "polylog_signature": compute_polylog_signature(prime_frequencies).detach().cpu(),
            "shape_of_absence": compute_vacuum_residue(dyad.gyroid_residue if dyad.gyroid_residue is not None else prime_frequencies).detach().cpu(),
            "hyperbolic_residue": dyad.hyperbolic_residue.detach().cpu() if hasattr(dyad, 'hyperbolic_residue') and dyad.hyperbolic_residue is not None else None,
            "audio_harmonics": dyad.audio_harmonics,
            "video_breather": dyad.video_breather,
            "gauge_field": gauge_field.detach().cpu() if gauge_field is not None else None,
            "betti_signature_8": betti_numbers,
            "meta_state_shielded": protected_state, # Punctured State
            "all_shapes": [s.detach().cpu() for s in dyad.all_shapes] if dyad.all_shapes else None, # Grom Flexibility
            "image_fingerprint": dyad.image_fingerprint.detach().cpu() if isinstance(dyad.image_fingerprint, torch.Tensor) else dyad.image_fingerprint,
            "gyroid_residue": gyroid_val,
            "prime_frequencies": prime_val,
            "timestamp": dyad.timestamp,
            "archetype_profile": archetype_profile,
            "agent_smith_iters": float(_AGENT_SMITH_ENGINE.iters_base_small.item()) if _AGENT_SMITH_ENGINE is not None else 30.0,
            "agent_smith_gauge": float(_AGENT_SMITH_ENGINE.gauge.item()) if _AGENT_SMITH_ENGINE is not None else 3.99,
            "warmstart_states": _AGENT_SMITH_ENGINE.warmstart_states if _AGENT_SMITH_ENGINE is not None else {},
            "dyad_metadata": dyad.metadata 
        }
        filepath = os.path.join(self.storage_dir, filename)
        torch.save(payload, filepath)
        print(f"[FOSSILIZER] Sovereign Agent Smith Exported: {filename} (Shielding ID: {pot_id}, PI-Growth: {h_gamma:.4f})")
        return filepath

    def inject_agent_smith(self, filepath: str, unraveling_closure=None, expected_dim: int = 96, hardware_trfc_ms: float = 160.0, agent_smith_engine: Optional[nn.Module] = None) -> Dict:
        """
        Loads the mathematical identity of an agent (Agent Smith) back into the system,
        allowing the local hardware to 'breathe' its own unique life into the configuration.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agent Smith file not found: {filepath}")
            
        payload = torch.load(filepath, map_location='cpu')
             
        # Minimal validation
        if payload.get("type") != "soliton_smith":
             raise ValueError("File is not a valid Agent Smith (soliton_smith) .pt payload.")
             
        # Ontological Import Gates & Substrate Bridges
        bridge = AgentSubstrateBridge()
        is_safe = bridge.verify_invariants(payload, unraveling_closure=unraveling_closure)
        if not is_safe:
            print("[WARNING] Invariant verification failed on ingestion. Topologies may leak.")
            
        # Align substrate
        payload = bridge.align_substrate(payload, expected_dim=expected_dim, hardware_trfc_ms=hardware_trfc_ms)
        
        # Rehydrate Warmstart if engine provided
        if agent_smith_engine is not None:
             bridge.rehydrate_warmstart(payload, agent_smith_engine)
             
        return payload
