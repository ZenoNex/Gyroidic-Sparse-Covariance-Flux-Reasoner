#!/usr/bin/env python3
"""
Enhanced Temporal Association Training with Proper Non-Lobotomy Architecture

This script demonstrates temporal association training using the proper
non-lobotomy architecture with polynomial co-prime functionals instead
of hardcoded primes.

Key Features:
- Polynomial Co-Prime Functionals (no hardcoded primes)
- Evolutionary Trust Selection
- Saturated Polynomial Gates with Bimodal Routing
- Proper Three-System Architecture (Horse/Horn/Magic)
- Non-teleological Flow
- Love Invariant Preservation

Author: William Matthew Bryant
Created: January 2026
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
import random
from dataclasses import dataclass
import json
from pathlib import Path

# Core imports - proper non-lobotomy architecture
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.polynomial_coprime import PolynomialCoprimeConfig, SaturatedPolynomialGate
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from src.core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from src.core.love_invariant_protector import LoveInvariantProtector, SoftSaturatedGates
from src.optimization.codes_driver import CODES


class NonLobotomyTemporalModel(nn.Module):
    """
    Temporal model following proper non-lobotomy architecture.
    
    Uses polynomial co-prime functionals instead of hardcoded primes,
    implements evolutionary trust selection, and maintains the three-system
    architecture (Horse/Horn/Magic).
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_functionals: int = 5,
        poly_degree: int = 4,
        device: str = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_functionals
        self.D = poly_degree + 1
        self.device = device
        
        # SYSTEM 1: The Intuitive Manifold (The "Horse")
        # Polynomial Co-Prime Functionals - NO HARDCODED PRIMES
        self.polynomial_config = PolynomialCoprimeConfig(
            k=num_functionals,
            degree=poly_degree,
            basis_type='chebyshev',
            learnable=True,
            use_saturation=True,
            device=device
        )
        
        # Core neural components
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # Bimodal Routing (Hard/Soft genome) - evolutionary selection
        self.register_buffer('bimodal_genome', torch.randint(0, 2, (self.K,), device=device))
        
        # Saturated Polynomial Gates
        self.saturated_gates = nn.ModuleList([
            SaturatedPolynomialGate() for _ in range(self.K)
        ])
        
        # SYSTEM 2: The Physical Constraint (The "Horn")
        # Repair system components
        self.spectral_corrector = SpectralCoherenceCorrector(device=device)
        self.bezout_refresh = BezoutCoefficientRefresh(self.K, poly_degree, device=device)
        self.chern_simons_gasket = ChernSimonsGasket(device=device)
        self.soliton_healer = SolitonStabilityHealer(device=device)
        
        # SYSTEM 3: "Dark Matter" (The "Magic")
        # Love Invariant and Chiral Glue
        self.love_protector = LoveInvariantProtector(hidden_dim, device=device)
        self.soft_gates = SoftSaturatedGates(self.K, poly_degree, device=device)
        
        # CODES Driver for proper PAS_h computation
        self.codes_driver = CODES(coherence_threshold=0.75)
        
        # Evolutionary Trust Selection (not fixed optimization)
        self.register_buffer('trust_scalars', torch.ones(self.K, device=device))
        self.register_buffer('mutation_strengths', torch.full((self.K,), 0.05, device=device))
        self.register_buffer('is_fossilized', torch.zeros(self.K, dtype=torch.bool, device=device))
        
        # Temporal state tracking for coherence
        self.register_buffer('prev_states', torch.zeros(3, hidden_dim, device=device))
        self.state_history_idx = 0
        
        # Pressure tracking for saturation detection
        self._pressure_history = {k: [] for k in range(self.K)}
        self.saturation_threshold = 0.05
        self.saturation_window = 20
    
    def forward(self, x: torch.Tensor, return_analysis: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass using proper non-lobotomy architecture.
        
        Follows the three-system interaction:
        1. System 1 (Horse): Polynomial functionals with bimodal routing
        2. System 2 (Horn): Physical constraint probes
        3. System 3 (Magic): Love invariant and fossilization
        """
        batch_size = x.shape[0]
        
        # Input projection
        h = torch.relu(self.input_proj(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        
        # SYSTEM 1: Polynomial Co-Prime Functionals
        # Evaluate polynomial functionals (NO HARDCODED PRIMES)
        # Create K different scalar projections from the hidden vector
        phi_values = torch.zeros(h.shape[0], self.K, device=self.device)
        
        # Use learned linear projections to create K scalar inputs
        if not hasattr(self, 'functional_projections'):
            self.functional_projections = nn.ModuleList([
                nn.Linear(h.shape[1], 1) for _ in range(self.K)
            ]).to(self.device, non_blocking=True)
        
        for k in range(self.K):
            # Project hidden vector to scalar for k-th functional
            h_k = self.functional_projections[k](h)  # [batch, 1]
            phi_k_full = self.polynomial_config.evaluate(h_k)  # [batch, 1, K]
            phi_values[:, k] = phi_k_full[:, 0, k]  # Take k-th functional value
        
        # Apply bimodal routing (evolutionary genome selection)
        routed_phi = torch.zeros_like(phi_values)
        for k in range(self.K):
            if self.bimodal_genome[k] == 0:
                # Soft mode: differentiable
                routed_phi[:, k] = torch.tanh(phi_values[:, k])
            else:
                # Hard mode: saturated
                routed_phi[:, k] = self.saturated_gates[k](phi_values[:, k])
        
        # SYSTEM 2: Physical Constraint Probes
        # Only invoke if containment pressure exceeded
        containment_pressure = self._compute_containment_pressure(routed_phi)
        
        if containment_pressure > 0.5:  # Rescue trigger
            # Spectral coherence correction
            h_corrected = self.spectral_corrector.adaptive_coherence_correction(h.unsqueeze(1))
            h = h_corrected.squeeze(1)
            
            # Bezout coefficient refresh (for CRT)
            routed_phi = self.bezout_refresh.apply_crt_correction(routed_phi.unsqueeze(1)).squeeze(1)
            
            # Chern-Simons gasket (plug logic leaks)
            # Use polynomial coefficients instead of prime indices
            poly_coeffs = self.polynomial_config.get_coefficients_tensor()
            routed_phi = self.chern_simons_gasket.plug_logic_leak(routed_phi.unsqueeze(1), poly_coeffs).squeeze(1)
            
            # Soliton stability healing
            routed_phi = self.soliton_healer.heal_fractured_soliton(routed_phi.unsqueeze(1)).squeeze(1)
        
        # SYSTEM 3: Love Invariant and Fossilization
        # Apply love protection (non-ownable flow)
        love_vector, love_diagnostics = self.love_protector.apply_love_protection(h)
        
        # Soft saturated gates with PAS_h
        pas_h = self._compute_pas_h(routed_phi)
        routed_phi = self.soft_gates.apply_soft_saturation(routed_phi.unsqueeze(1), pas_h).squeeze(1)
        
        # CRT Reconstruction using polynomial functionals
        reconstructed = self._polynomial_crt_reconstruction(routed_phi)
        
        # Output (simple for now)
        output = reconstructed.mean(dim=1)
        
        # Update temporal state history
        self._update_state_history(h.detach().mean(dim=0))
        
        # Update pressure history for evolutionary selection
        for k in range(self.K):
            pressure = self._compute_functional_pressure(routed_phi[:, k])
            self._update_pressure_history(k, pressure)
        
        results = {
            'output': output,
            'hidden_state': h,
            'phi_values': routed_phi,
            'reconstructed': reconstructed,
            'trust_scalars': self.trust_scalars.clone(),
            'containment_pressure': containment_pressure,
            'pas_h': pas_h
        }
        
        if return_analysis:
            results.update({
                'polynomial_diagnostics': self._get_polynomial_diagnostics(),
                'bimodal_genome': self.bimodal_genome.clone(),
                'fossilization_status': self.is_fossilized.clone(),
                'orthogonality_pressure': self.polynomial_config.orthogonality_pressure(),
                'coprimality_pressure': self.polynomial_config.co_primality_pressure(),
                'love_diagnostics': love_diagnostics,
                'spectral_diagnostics': self.spectral_corrector.get_diagnostics() if hasattr(self.spectral_corrector, 'get_diagnostics') else {},
                'saturation_status': self._get_saturation_status()
            })
        
        return results
    
    def _compute_containment_pressure(self, phi: torch.Tensor) -> float:
        """Compute containment pressure to trigger System 2."""
        # Simple heuristic: high variance indicates instability
        variance = phi.var().item()
        return min(variance / 2.0, 1.0)
    
    def _compute_pas_h(self, phi: torch.Tensor) -> float:
        """
        Compute Phase Alignment Score using proper CODES multiharmonic alignment.
        
        PAS_h = Σ_{d=0}^D (1/(d+1)) * ||θ_d||_2
        
        This follows the INVARIANT_OPTIMIZATION.md specification for proper
        harmonic phase alignment scoring.
        """
        # Get polynomial coefficients tensor [K, D]
        theta = self.polynomial_config.get_coefficients_tensor()
        
        # Compute multiharmonic phase alignment score
        pas_h = 0.0
        D = theta.shape[1]  # Polynomial degree + 1
        
        for d in range(D):
            # Harmonic weight: 1/(d+1) - higher weight for lower degrees (fundamental modes)
            harmonic_weight = 1.0 / (d + 1)
            
            # L2 norm of degree-d coefficients across all functionals
            theta_d_norm = torch.norm(theta[:, d]).item()
            
            # Weighted contribution
            pas_h += harmonic_weight * theta_d_norm
        
        # Use CODES driver for additional phase coherence computation
        # Derive phase from phi tensor statistics
        phi_phase = float(torch.sum(phi).item() % (2 * math.pi))
        codes_coherence = self.codes_driver.compute_pas_h(phi_phase)
        
        # Combine polynomial harmonic alignment with CODES coherence
        # Higher values indicate better phase alignment
        combined_pas_h = 0.7 * pas_h + 0.3 * codes_coherence
        
        return combined_pas_h
    
    def _polynomial_crt_reconstruction(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Chinese Remainder Theorem reconstruction using polynomial functionals.
        
        Instead of using prime moduli, we use the polynomial basis coefficients
        as the reconstruction weights.
        """
        # Handle variable tensor dimensions
        if phi.dim() == 2:
            batch_size, K = phi.shape
        elif phi.dim() == 3:
            batch_size, K, _ = phi.shape
            phi = phi.mean(dim=-1)  # Average over last dimension if present
        else:
            raise ValueError(f"Unexpected phi tensor dimensions: {phi.shape}")
        
        # Get polynomial coefficients as reconstruction weights
        theta = self.polynomial_config.get_coefficients_tensor()  # [K, D]
        
        # Weighted reconstruction
        reconstructed = torch.zeros(batch_size, self.D, device=self.device)
        
        for k in range(K):
            # Weight by polynomial coefficients
            contribution = phi[:, k:k+1] * theta[k:k+1, :]  # [batch, 1] * [1, D] = [batch, D]
            reconstructed += contribution
        
        return reconstructed
    
    def _compute_functional_pressure(self, phi_k: torch.Tensor) -> float:
        """Compute pressure for functional k."""
        # BOSTICK ADAPTATION: Prevent 0.30 collapse by ensuring non-zero degrees of freedom
        if phi_k.numel() <= 1:
            return 1e-4 # Minimum baseline flux

        # Calculate variance on detached tensor to stop UI drag
        v = phi_k.detach().var()

        # If variance is zero or NaN, provide a 'fossilized' baseline
        if torch.isnan(v) or v < 1e-9:
            return 1e-4

        return (v + 1e-8).item()
    
    def _update_pressure_history(self, k: int, pressure: float):
        """Update pressure history for saturation detection."""
        if k not in self._pressure_history:
            self._pressure_history[k] = []
        self._pressure_history[k].append(pressure)
        
        # Bounded memory
        if len(self._pressure_history[k]) > self.saturation_window * 2:
            self._pressure_history[k] = self._pressure_history[k][-self.saturation_window:]
    
    def _is_saturated(self, k: int) -> bool:
        """Check if functional k has reached constraint geometry saturation."""
        history = self._pressure_history.get(k, [])
        
        if len(history) < self.saturation_window:
            return False
        
        recent = torch.tensor(history[-self.saturation_window:])
        oscillation = recent.std()
        
        return oscillation.item() < self.saturation_threshold
    
    def _get_saturation_status(self) -> Dict[int, bool]:
        """Get saturation status for all functionals."""
        return {k: self._is_saturated(k) for k in range(self.K)}
    
    def _update_state_history(self, new_state: torch.Tensor):
        """Update temporal state history."""
        self.prev_states[self.state_history_idx] = new_state
        self.state_history_idx = (self.state_history_idx + 1) % 3
    
    def _get_polynomial_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics for polynomial system."""
        theta = self.polynomial_config.get_coefficients_tensor()
        return {
            'coefficient_norm': torch.norm(theta).item(),
            'coefficient_rank': torch.linalg.matrix_rank(theta).item(),
            'chirality_preserved': self._check_chirality(),
            'birkhoff_constraint_satisfied': self._check_birkhoff_constraints(theta)
        }
    
    def _check_chirality(self) -> bool:
        """Check if chirality is preserved (non-zero asymmetry)."""
        theta = self.polynomial_config.get_coefficients_tensor()
        
        # Check parity mixing
        even_mask = torch.arange(self.D, device=self.device) % 2 == 0
        odd_mask = ~even_mask
        
        even_energy = (theta[:, even_mask] ** 2).sum(dim=1)
        odd_energy = (theta[:, odd_mask] ** 2).sum(dim=1)
        
        # Chirality requires mixing of parities
        pure_even = odd_energy < 1e-6
        pure_odd = even_energy < 1e-6
        symmetric_defect = pure_even | pure_odd
        
        return not symmetric_defect.any()
    
    def _check_birkhoff_constraints(self, theta: torch.Tensor) -> bool:
        """Check if Birkhoff polytope constraints are satisfied."""
        # Row sums should be approximately 1
        row_sums = theta.sum(dim=1)
        row_constraint = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
        
        # Column sums should be approximately 1 (if square)
        if theta.shape[0] == theta.shape[1]:
            col_sums = theta.sum(dim=0)
            col_constraint = torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-2)
        else:
            col_constraint = True  # Skip for non-square matrices
        
        # All entries should be non-negative
        non_negative = (theta >= -1e-6).all()
        
        return row_constraint and col_constraint and non_negative
    
    def evolve_system(self):
        """
        Evolutionary step: mutate non-fossilized functionals.
        
        This implements evolutionary trust selection instead of
        fixed optimization.
        """
        # Mutate polynomial coefficients
        self.polynomial_config.mutate()
        
        # Evolve bimodal genome
        active_mask = ~self.is_fossilized
        if active_mask.any():
            # Random genome mutations
            mutation_prob = 0.1
            mutations = torch.rand(self.K, device=self.device) < mutation_prob
            mutation_mask = active_mask & mutations
            
            if mutation_mask.any():
                self.bimodal_genome[mutation_mask] = 1 - self.bimodal_genome[mutation_mask]
    
    def attempt_fossilization(self):
        """
        Attempt to fossilize saturated functionals.
        
        Only fossilizes at admissibility boundaries, not during
        active saturation (prevents premature topology lock-in).
        """
        fossilization_events = []
        
        for k in range(self.K):
            if not self.is_fossilized[k] and self._is_saturated(k):
                # Check if trust is high enough
                if self.trust_scalars[k] > 0.8:
                    self.is_fossilized[k] = True
                    fossilization_events.append(k)
        
        return fossilization_events


class NonLobotomyTemporalTrainer:
    """
    Trainer following non-lobotomy principles.
    
    Uses evolutionary trust selection instead of gradient descent
    on trust scalars. Implements proper survivorship pressure.
    """
    
    def __init__(
        self,
        model: NonLobotomyTemporalModel,
        dataset,
        evolution_rate: float = 0.02,
        survivorship_threshold: float = 0.7
    ):
        self.model = model
        self.dataset = dataset
        self.evolution_rate = evolution_rate
        self.survivorship_threshold = survivorship_threshold
        
        # Optimizer for neural components only
        # Polynomial coefficients evolve via mutation, not gradient descent
        neural_params = []
        for name, param in model.named_parameters():
            if 'polynomial_config' not in name:
                neural_params.append(param)
        
        self.optimizer = torch.optim.Adam(neural_params, lr=1e-4)
        
        # Training history
        self.history = {
            'survivorship_pressure': [],
            'association_accuracy': [],
            'temporal_coherence': [],
            'trust_evolution': [],
            'fossilization_events': [],
            'evolutionary_steps': []
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with survivorship pressure."""
        sequences = batch['sequences']  # [batch, seq_len, embed_dim]
        targets = batch['targets']      # [batch, seq_len, embed_dim]
        
        batch_size, seq_len, embed_dim = sequences.shape
        
        total_survivorship_pressure = 0.0
        coherence_scores = []
        association_accuracies = []
        
        # Process sequence
        for t in range(seq_len):
            current_input = sequences[:, t, :]
            current_target = targets[:, t, :]
            
            # Forward pass
            output = self.model(current_input, return_analysis=True)
            
            # Association accuracy (not loss - we measure survivorship)
            hidden_state = output['hidden_state']
            
            # Project target to hidden dimension
            if current_target.shape[1] != hidden_state.shape[1]:
                target_proj = F.adaptive_avg_pool1d(
                    current_target.unsqueeze(1), 
                    hidden_state.shape[1]
                ).squeeze(1)
            else:
                target_proj = current_target
            
            association_accuracy = F.cosine_similarity(hidden_state, target_proj, dim=1).mean()
            association_accuracies.append(association_accuracy)
            
            # Temporal coherence
            coherence = self._compute_temporal_coherence(hidden_state)
            coherence_scores.append(coherence)
            
            # Survivorship pressure (not loss)
            # Higher pressure = lower survivorship
            survivorship_pressure = 1.0 - association_accuracy + 0.1 * (1.0 - coherence)
            total_survivorship_pressure += survivorship_pressure
        
        # Average metrics
        avg_survivorship_pressure = total_survivorship_pressure / seq_len
        avg_association_accuracy = torch.stack(association_accuracies).mean()
        avg_coherence = torch.stack(coherence_scores).mean()
        
        # Neural component optimization (not polynomial coefficients)
        self.optimizer.zero_grad()
        avg_survivorship_pressure.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Evolutionary trust update
        self._update_trust_evolutionary(avg_association_accuracy.item(), avg_coherence.item())
        
        return {
            'survivorship_pressure': avg_survivorship_pressure.item(),
            'association_accuracy': avg_association_accuracy.item(),
            'temporal_coherence': avg_coherence.item(),
            'trust_mean': self.model.trust_scalars.mean().item(),
            'trust_std': self.model.trust_scalars.std().item(),
            'num_fossilized': self.model.is_fossilized.sum().item(),
            'containment_pressure': output['containment_pressure'],
            'pas_h': output['pas_h']
        }
    
    def _compute_temporal_coherence(self, current_state: torch.Tensor) -> torch.Tensor:
        """Compute temporal coherence using state history."""
        if torch.allclose(self.model.prev_states, torch.zeros_like(self.model.prev_states)):
            return torch.tensor(1.0, device=self.model.device)
        
        coherences = []
        current_mean = current_state.mean(dim=0)
        
        for i in range(3):
            if not torch.allclose(self.model.prev_states[i], torch.zeros_like(self.model.prev_states[i])):
                similarity = F.cosine_similarity(
                    current_mean.unsqueeze(0), 
                    self.model.prev_states[i].unsqueeze(0), 
                    dim=1
                )
                coherences.append(similarity)
        
        if coherences:
            return torch.stack(coherences).mean()
        else:
            return torch.tensor(1.0, device=self.model.device)
    
    def _update_trust_evolutionary(self, association_accuracy: float, coherence: float):
        """
        Update trust using evolutionary selection, not gradient descent.
        
        Trust evolves based on survivorship, not optimization.
        """
        # Combined performance score
        performance = 0.7 * association_accuracy + 0.3 * coherence
        
        # Evolutionary pressure
        if performance > self.survivorship_threshold:
            # Increase trust for survivors
            trust_delta = self.evolution_rate * (performance - self.survivorship_threshold)
            self.model.trust_scalars += trust_delta
        else:
            # Decrease trust for non-survivors
            trust_delta = self.evolution_rate * (performance - self.survivorship_threshold)
            self.model.trust_scalars += trust_delta
        
        # Clamp trust
        self.model.trust_scalars.clamp_(0.0, 1.0)
    
    def train_epoch(self, num_batches: int = 30) -> Dict[str, float]:
        """Train for one epoch with evolutionary steps."""
        epoch_metrics = []
        
        for batch_idx in range(num_batches):
            # Generate batch
            batch = self.dataset.get_batch(batch_size=4)
            
            # Train step
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
            
            # Evolutionary step every 10 batches
            if batch_idx % 10 == 0:
                self.model.evolve_system()
                fossilization_events = self.model.attempt_fossilization()
                
                if fossilization_events:
                    print(f"🔒 Fossilized functionals: {fossilization_events}")
                    self.history['fossilization_events'].extend(fossilization_events)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:2d}: "
                      f"Assoc={metrics['association_accuracy']:.3f}, "
                      f"Coherence={metrics['temporal_coherence']:.3f}, "
                      f"Trust={metrics['trust_mean']:.3f}±{metrics['trust_std']:.3f}, "
                      f"PAS_h={metrics['pas_h']:.3f}")
        
        # Compute epoch averages
        epoch_summary = {}
        for key in ['survivorship_pressure', 'association_accuracy', 'temporal_coherence', 'trust_mean', 'trust_std', 'containment_pressure', 'pas_h']:
            epoch_summary[key] = np.mean([m[key] for m in epoch_metrics])
        
        epoch_summary['final_num_fossilized'] = epoch_metrics[-1]['num_fossilized']
        
        # Update history
        for key in ['survivorship_pressure', 'association_accuracy', 'temporal_coherence']:
            self.history[key].append(epoch_summary[key])
        
        return epoch_summary


class SimpleTemporalDataset:
    """Simple temporal dataset for testing."""
    
    def __init__(self, sequence_length: int = 8, num_concepts: int = 50, embedding_dim: int = 768, device: str = None):
        self.sequence_length = sequence_length
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Create concept embeddings
        self.concept_embeddings = torch.randn(num_concepts, embedding_dim, device=device)
        
        # Create associations (clusters)
        self.associations = self._create_associations()
    
    def _create_associations(self):
        """Create temporal associations between concepts."""
        associations = {}
        cluster_size = 10
        num_clusters = self.num_concepts // cluster_size
        
        for cluster_id in range(num_clusters):
            cluster_start = cluster_id * cluster_size
            cluster_end = cluster_start + cluster_size
            
            for concept_id in range(cluster_start, cluster_end):
                # Associate with other concepts in the same cluster
                cluster_concepts = list(range(cluster_start, cluster_end))
                cluster_concepts.remove(concept_id)
                associations[concept_id] = cluster_concepts[:5]  # Top 5 associations
        
        return associations
    
    def get_batch(self, batch_size: int = 4):
        """Generate a batch of temporal sequences."""
        sequences = []
        targets = []
        
        for _ in range(batch_size):
            sequence = []
            sequence_targets = []
            
            # Start with random concept
            current_concept = np.random.randint(self.num_concepts)
            
            for step in range(self.sequence_length):
                # Add current concept to sequence
                sequence.append(self.concept_embeddings[current_concept])
                
                # Target is associated concept
                if current_concept in self.associations:
                    target_concepts = self.associations[current_concept]
                    target_concept = np.random.choice(target_concepts)
                    target_embedding = self.concept_embeddings[target_concept]
                else:
                    target_embedding = self.concept_embeddings[current_concept]
                
                sequence_targets.append(target_embedding)
                
                # Move to next concept (with some probability)
                if current_concept in self.associations and np.random.random() > 0.3:
                    current_concept = np.random.choice(self.associations[current_concept])
                else:
                    current_concept = np.random.randint(self.num_concepts)
            
            sequences.append(torch.stack(sequence))
            targets.append(torch.stack(sequence_targets))
        
        return {
            'sequences': torch.stack(sequences),
            'targets': torch.stack(targets)
        }


def run_non_lobotomy_temporal_training():
    """Run temporal training with proper non-lobotomy architecture."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Non-Lobotomy Temporal Association Training")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create model with proper architecture
    print("🏗️ Creating non-lobotomy model...")
    model = NonLobotomyTemporalModel(
        input_dim=768,
        hidden_dim=256,
        num_functionals=5,
        poly_degree=4,
        device=device
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Polynomial functionals: {model.K}")
    print(f"   Polynomial degree: {model.D - 1}")
    print(f"   Bimodal genome: {model.bimodal_genome.tolist()}")
    print(f"   Trust scalars: {[f'{t:.3f}' for t in model.trust_scalars.tolist()]}")
    
    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = SimpleTemporalDataset(sequence_length=6, num_concepts=30, device=device)
    print(f"✅ Dataset created with {dataset.num_concepts} concepts")
    
    # Create trainer
    print("\n🎯 Creating trainer...")
    trainer = NonLobotomyTemporalTrainer(model, dataset)
    print("✅ Trainer created")
    
    # Test functionality
    print("\n🧪 Testing functionality...")
    sample_batch = dataset.get_batch(batch_size=2)
    print(f"   Sequences shape: {sample_batch['sequences'].shape}")
    print(f"   Targets shape: {sample_batch['targets'].shape}")
    
    with torch.no_grad():
        test_output = model(sample_batch['sequences'][0, 0, :].unsqueeze(0), return_analysis=True)
        print(f"   Test output shape: {test_output['output'].shape}")
        print(f"   Polynomial diagnostics: {test_output['polynomial_diagnostics']}")
        print(f"   Orthogonality pressure keys: {list(test_output['orthogonality_pressure'].keys())}")
    
    # Training loop
    num_epochs = 5
    batches_per_epoch = 20
    
    print(f"\n🚀 Starting training: {num_epochs} epochs, {batches_per_epoch} batches each")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        
        try:
            epoch_metrics = trainer.train_epoch(num_batches=batches_per_epoch)
            
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Survivorship Pressure: {epoch_metrics['survivorship_pressure']:.3f}")
            print(f"   Association Accuracy: {epoch_metrics['association_accuracy']:.3f}")
            print(f"   Temporal Coherence: {epoch_metrics['temporal_coherence']:.3f}")
            print(f"   Trust Mean: {epoch_metrics['trust_mean']:.3f} ± {epoch_metrics['trust_std']:.3f}")
            print(f"   Fossilized: {epoch_metrics['final_num_fossilized']}")
            print(f"   Containment Pressure: {epoch_metrics['containment_pressure']:.3f}")
            print(f"   PAS_h: {epoch_metrics['pas_h']:.3f}")
            
            current_trust = model.trust_scalars
            print(f"   Trust Scalars: {[f'{t:.3f}' for t in current_trust.tolist()]}")
            
        except Exception as e:
            print(f"❌ Epoch {epoch + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n🎯 Training Complete!")
    print(f"Final Trust: {[f'{t:.3f}' for t in model.trust_scalars.tolist()]}")
    print(f"Final Bimodal Genome: {model.bimodal_genome.tolist()}")
    print(f"Fossilization Events: {len(trainer.history['fossilization_events'])}")
    
    return model, trainer


if __name__ == "__main__":
    print("🧠 Non-Lobotomy Temporal Association Training")
    print("Using polynomial co-prime functionals and evolutionary trust selection")
    print("=" * 75)
    
    try:
        model, trainer = run_non_lobotomy_temporal_training()
        print(f"\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

