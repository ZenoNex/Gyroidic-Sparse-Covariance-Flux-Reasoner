#!/usr/bin/env python3
"""
Enhanced Temporal Association Training with Proper Non-Lobotomy Architecture

This module implements temporal association training using the proper
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
Ported to src/training: May 2026
"""

import sys
import os

# Ensure workspace root is in sys.path for reliable src module imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

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

from src.core.honest_jitter import harvest_honest_jitter
from src.core.polynomial_coprime import PolynomialCoprimeConfig, SaturatedPolynomialGate
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from src.core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from src.core.love_invariant_protector import LoveInvariantProtector, SoftSaturatedGates
from src.optimization.codes_driver import CODES
from src.optimization.ricci_flow_optimizer import RicciFlowOptimizer, BouligandWillmoreGasket
from src.core.birkhoff_projection import BouligandBirkhoffProjectionFunction


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
        
        # Functional projections for System 1 scalar evaluation
        self.functional_projections = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(self.K)
        ]).to(self.device, non_blocking=True)
        
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
        phi_values = torch.zeros(h.shape[0], self.K, device=self.device)
        
        for k in range(self.K):
            h_k = self.functional_projections[k](h)
            phi_k_full = self.polynomial_config.evaluate(h_k)
            phi_values[:, k] = phi_k_full[:, 0, k]
        
        # Apply bimodal routing
        routed_phi = torch.zeros_like(phi_values)
        for k in range(self.K):
            if self.bimodal_genome[k] == 0:
                routed_phi[:, k] = torch.tanh(phi_values[:, k])
            else:
                routed_phi[:, k] = self.saturated_gates[k](phi_values[:, k])
        
        # SYSTEM 2: Physical Constraint Probes
        containment_pressure = self._compute_containment_pressure(routed_phi)
        
        if containment_pressure > 0.5:
            h_corrected = self.spectral_corrector.adaptive_coherence_correction(h.unsqueeze(1))
            h = h_corrected.squeeze(1)
            routed_phi = self.bezout_refresh.apply_crt_correction(routed_phi.unsqueeze(1)).squeeze(1)
            poly_coeffs = self.polynomial_config.get_coefficients_tensor()
            routed_phi = self.chern_simons_gasket.plug_logic_leak(routed_phi.unsqueeze(1), poly_coeffs).squeeze(1)
            routed_phi = self.soliton_healer.heal_fractured_soliton(routed_phi.unsqueeze(1)).squeeze(1)
        
        # SYSTEM 3: Love Invariant and Fossilization
        love_vector, love_diagnostics = self.love_protector.apply_love_protection(h)
        pas_h = self._compute_pas_h(routed_phi)
        routed_phi = self.soft_gates.apply_soft_saturation(routed_phi.unsqueeze(1), pas_h).squeeze(1)
        reconstructed = self._polynomial_crt_reconstruction(routed_phi)
        output = reconstructed.mean(dim=1)
        
        self._update_state_history(h.detach().mean(dim=0))
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
        variance = phi.var().item()
        return min(variance / 2.0, 1.0)
    
    def _compute_pas_h(self, phi: torch.Tensor) -> float:
        theta = self.polynomial_config.get_coefficients_tensor()
        pas_h = 0.0
        D = theta.shape[1]
        for d in range(D):
            harmonic_weight = 1.0 / (d + 1)
            theta_d_norm = torch.norm(theta[:, d]).item()
            pas_h += harmonic_weight * theta_d_norm
        
        phi_phase = float(torch.sum(phi).item() % (2 * math.pi))
        codes_coherence = self.codes_driver.compute_pas_h(phi_phase)
        return 0.7 * pas_h + 0.3 * codes_coherence
    
    def _polynomial_crt_reconstruction(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 2:
            batch_size, K = phi.shape
        elif phi.dim() == 3:
            batch_size, K, _ = phi.shape
            phi = phi.mean(dim=-1)
        else:
            raise ValueError(f"Unexpected phi tensor dimensions: {phi.shape}")
        
        theta = self.polynomial_config.get_coefficients_tensor()
        reconstructed = torch.zeros(batch_size, self.D, device=self.device)
        
        for k in range(K):
            contribution = phi[:, k:k+1] * theta[k:k+1, :]
            reconstructed += contribution
        return reconstructed
    
    def _compute_functional_pressure(self, phi_k: torch.Tensor) -> float:
        if phi_k.numel() <= 1:
            return 1e-4
        v = phi_k.detach().var()
        if torch.isnan(v) or v < 1e-9:
            return 1e-4
        return (v + 1e-8).item()
    
    def _update_pressure_history(self, k: int, pressure: float):
        if k not in self._pressure_history:
            self._pressure_history[k] = []
        self._pressure_history[k].append(pressure)
        if len(self._pressure_history[k]) > self.saturation_window * 2:
            self._pressure_history[k] = self._pressure_history[k][-self.saturation_window:]
    
    def _is_saturated(self, k: int) -> bool:
        history = self._pressure_history.get(k, [])
        if len(history) < self.saturation_window:
            return False
        recent = torch.tensor(history[-self.saturation_window:])
        oscillation = recent.std()
        return oscillation.item() < self.saturation_threshold
    
    def _get_saturation_status(self) -> Dict[int, bool]:
        return {k: self._is_saturated(k) for k in range(self.K)}
    
    def _update_state_history(self, new_state: torch.Tensor):
        self.prev_states[self.state_history_idx] = new_state
        self.state_history_idx = (self.state_history_idx + 1) % 3
    
    def _get_polynomial_diagnostics(self) -> Dict[str, Any]:
        theta = self.polynomial_config.get_coefficients_tensor()
        return {
            'coefficient_norm': torch.norm(theta).item(),
            'coefficient_rank': torch.linalg.matrix_rank(theta).item(),
            'chirality_preserved': self._check_chirality(),
            'birkhoff_constraint_satisfied': self._check_birkhoff_constraints(theta)
        }
    
    def _check_chirality(self) -> bool:
        theta = self.polynomial_config.get_coefficients_tensor()
        even_mask = torch.arange(self.D, device=self.device) % 2 == 0
        odd_mask = ~even_mask
        even_energy = (theta[:, even_mask] ** 2).sum(dim=1)
        odd_energy = (theta[:, odd_mask] ** 2).sum(dim=1)
        pure_even = odd_energy < 1e-6
        pure_odd = even_energy < 1e-6
        symmetric_defect = pure_even | pure_odd
        return not symmetric_defect.any()
    
    def _check_birkhoff_constraints(self, theta: torch.Tensor) -> bool:
        row_sums = theta.sum(dim=1)
        row_constraint = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
        if theta.shape[0] == theta.shape[1]:
            col_sums = theta.sum(dim=0)
            col_constraint = torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-2)
        else:
            col_constraint = True
        non_negative = (theta >= -1e-6).all()
        return row_constraint and col_constraint and non_negative
    
    def evolve_system(self):
        self.polynomial_config.mutate()
        active_mask = ~self.is_fossilized
        if active_mask.any():
            mutation_prob = 0.1
            _jitter = harvest_honest_jitter((self.K,), device=self.device, scaled=False)
            _rand_vals = (_jitter + 1.0) / 2.0
            mutations = _rand_vals < mutation_prob
            mutation_mask = active_mask & mutations
            if mutation_mask.any():
                self.bimodal_genome[mutation_mask] = 1 - self.bimodal_genome[mutation_mask]
    
    def attempt_fossilization(self):
        fossilization_events = []
        
        # [ARCHITECTURAL REMEDIATION] Replaced naive `trust > 0.8` fossilization bypass
        # with the formal DAQUFOperator to evaluate structural persistence based on
        # unknowledge contradiction load and speculative flux bounds.
        from src.core.daqf_operator import DAQUFOperator
        if not hasattr(self, '_daquf_operator'):
            self._daquf_operator = DAQUFOperator(num_fossils=self.K, fossil_dim=1, device=self.device)
            
        flux_scores = self.trust_scalars.unsqueeze(1)
        persistence = self._daquf_operator.speculate_persistence(flux_scores)
        
        for k in range(self.K):
            if not self.is_fossilized[k] and self._is_saturated(k):
                # We enforce fossilization only if DAQUF formally grants persistence
                if persistence[k] > 0:
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
        
        neural_params = []
        for name, param in model.named_parameters():
            if 'polynomial_config' not in name:
                neural_params.append(param)
        
        # [ANTI-LOBOTOMY ENFORCEMENT] Replace Adam with Ricci Flow
        self.optimizer = RicciFlowOptimizer(neural_params, lr=1e-3, seam_width=0.1)
        self.willmore_energy = BouligandWillmoreGasket()
        
        self.history = {
            'survivorship_pressure': [],
            'association_accuracy': [],
            'temporal_coherence': [],
            'trust_evolution': [],
            'fossilization_events': [],
            'evolutionary_steps': []
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        sequences = batch['sequences']
        targets = batch['targets']
        
        batch_size, seq_len, embed_dim = sequences.shape
        total_survivorship_pressure = 0.0
        coherence_scores = []
        association_accuracies = []
        
        for t in range(seq_len):
            current_input = sequences[:, t, :]
            current_target = targets[:, t, :]
            output = self.model(current_input, return_analysis=True)
            hidden_state = output['hidden_state']
            
            if current_target.shape[1] != hidden_state.shape[1]:
                target_proj = F.adaptive_avg_pool1d(
                    current_target.unsqueeze(1), 
                    hidden_state.shape[1]
                ).squeeze(1)
            else:
                target_proj = current_target
            
            association_accuracy = F.cosine_similarity(hidden_state, target_proj, dim=1).mean()
            association_accuracies.append(association_accuracy)
            
            coherence = self._compute_temporal_coherence(hidden_state)
            coherence_scores.append(coherence)
            
            survivorship_pressure = 1.0 - association_accuracy + 0.1 * (1.0 - coherence)
            total_survivorship_pressure += survivorship_pressure
        
        avg_survivorship_pressure = total_survivorship_pressure / seq_len
        avg_association_accuracy = torch.stack(association_accuracies).mean()
        avg_coherence = torch.stack(coherence_scores).mean()
        
        self.optimizer.zero_grad()
        
        # [ANTI-LOBOTOMY ENFORCEMENT] Calculate structural tension instead of scalar MSE
        structural_tension = self.willmore_energy(hidden_state) + avg_survivorship_pressure
        
        # Backpropagate topological stress
        structural_tension.backward()
        
        # [BOULIGAND PROJECTION] Apply Bouligand projections on gradients if needed via manifold
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
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
        return torch.stack(coherences).mean() if coherences else torch.tensor(1.0, device=self.model.device)
    
    def _update_trust_evolutionary(self, association_accuracy: float, coherence: float):
        performance = 0.7 * association_accuracy + 0.3 * coherence
        if performance > self.survivorship_threshold:
            trust_delta = self.evolution_rate * (performance - self.survivorship_threshold)
            self.model.trust_scalars += trust_delta
        else:
            trust_delta = self.evolution_rate * (performance - self.survivorship_threshold)
            self.model.trust_scalars += trust_delta
        self.model.trust_scalars.clamp_(0.0, 1.0)
    
    def train_epoch(self, num_batches: int = 30) -> Dict[str, float]:
        epoch_metrics = []
        for batch_idx in range(num_batches):
            batch = self.dataset.get_batch(batch_size=4)
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
            
            if batch_idx % 10 == 0:
                self.model.evolve_system()
                fossilization_events = self.model.attempt_fossilization()
                if fossilization_events:
                    print(f"[LOCK] Fossilized functionals: {fossilization_events}")
                    self.history['fossilization_events'].extend(fossilization_events)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:2d}: "
                      f"Assoc={metrics['association_accuracy']:.3f}, "
                      f"Coherence={metrics['temporal_coherence']:.3f}, "
                      f"Trust={metrics['trust_mean']:.3f}±{metrics['trust_std']:.3f}, "
                      f"PAS_h={metrics['pas_h']:.3f}")
        
        epoch_summary = {}
        for key in ['survivorship_pressure', 'association_accuracy', 'temporal_coherence', 'trust_mean', 'trust_std', 'containment_pressure', 'pas_h']:
            epoch_summary[key] = np.mean([m[key] for m in epoch_metrics])
        epoch_summary['final_num_fossilized'] = epoch_metrics[-1]['num_fossilized']
        
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
        self.concept_embeddings = harvest_honest_jitter((num_concepts, embedding_dim), device=device, scaled=False)
        self.associations = self._create_associations()
    
    def _create_associations(self):
        associations = {}
        cluster_size = 10
        num_clusters = self.num_concepts // cluster_size
        for cluster_id in range(num_clusters):
            cluster_start = cluster_id * cluster_size
            cluster_end = cluster_start + cluster_size
            for concept_id in range(cluster_start, cluster_end):
                cluster_concepts = list(range(cluster_start, cluster_end))
                cluster_concepts.remove(concept_id)
                associations[concept_id] = cluster_concepts[:5]
        return associations
    
    def get_batch(self, batch_size: int = 4):
        sequences = []
        targets = []
        for _ in range(batch_size):
            sequence = []
            sequence_targets = []
            _j1 = (harvest_honest_jitter((1,), scaled=False).cpu().item() + 1.0) / 2.0
            current_concept = int(_j1 * self.num_concepts)
            for step in range(self.sequence_length):
                sequence.append(self.concept_embeddings[current_concept])
                if current_concept in self.associations:
                    target_concepts = self.associations[current_concept]
                    _j2 = (harvest_honest_jitter((1,), scaled=False).cpu().item() + 1.0) / 2.0
                    target_concept = target_concepts[int(_j2 * len(target_concepts))]
                    target_embedding = self.concept_embeddings[target_concept]
                else:
                    target_embedding = self.concept_embeddings[current_concept]
                sequence_targets.append(target_embedding)
                
                _j3 = (harvest_honest_jitter((1,), scaled=False).cpu().item() + 1.0) / 2.0
                if current_concept in self.associations and _j3 > 0.3:
                    _j4 = (harvest_honest_jitter((1,), scaled=False).cpu().item() + 1.0) / 2.0
                    current_concept = self.associations[current_concept][int(_j4 * len(self.associations[current_concept]))]
                else:
                    _j5 = (harvest_honest_jitter((1,), scaled=False).cpu().item() + 1.0) / 2.0
                    current_concept = int(_j5 * self.num_concepts)
            sequences.append(torch.stack(sequence))
            targets.append(torch.stack(sequence_targets))
        return {
            'sequences': torch.stack(sequences),
            'targets': torch.stack(targets)
        }


def run_non_lobotomy_temporal_training():
    """Run temporal training with proper non-lobotomy architecture."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[BRAIN] Non-Lobotomy Temporal Association Training")
    print(f"Device: {device}")
    print("=" * 60)
    
    print("[BUILD] Creating non-lobotomy model...")
    model = NonLobotomyTemporalModel(
        input_dim=768,
        hidden_dim=256,
        num_functionals=5,
        poly_degree=4,
        device=device
    )
    
    print(f"[OK] Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Polynomial functionals: {model.K}")
    print(f"   Polynomial degree: {model.D - 1}")
    
    print("\n[METRICS] Creating dataset...")
    dataset = SimpleTemporalDataset(sequence_length=6, num_concepts=30, device=device)
    print(f"[OK] Dataset created")
    
    print("\n[GOAL] Creating trainer...")
    trainer = NonLobotomyTemporalTrainer(model, dataset)
    print("[OK] Trainer created")
    
    print("\n[TEST] Testing functionality...")
    sample_batch = dataset.get_batch(batch_size=2)
    
    with torch.no_grad():
        test_output = model(sample_batch['sequences'][0, 0, :].unsqueeze(0), return_analysis=True)
        print(f"   Test output shape: {test_output['output'].shape}")
        print(f"   Polynomial diagnostics: {test_output['polynomial_diagnostics']}")
    
    num_epochs = 5
    batches_per_epoch = 20
    
    print(f"\n[START] Starting training: {num_epochs} epochs, {batches_per_epoch} batches each")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\n[DOCS] Epoch {epoch + 1}/{num_epochs}")
        try:
            epoch_metrics = trainer.train_epoch(num_batches=batches_per_epoch)
            print(f"\n[METRICS] Epoch {epoch + 1} Summary:")
            print(f"   Survivorship Pressure: {epoch_metrics['survivorship_pressure']:.3f}")
            print(f"   Association Accuracy: {epoch_metrics['association_accuracy']:.3f}")
            print(f"   Temporal Coherence: {epoch_metrics['temporal_coherence']:.3f}")
            print(f"   PAS_h: {epoch_metrics['pas_h']:.3f}")
        except Exception as e:
            print(f"[ERR] Epoch {epoch + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            break
            
    print(f"\n[GOAL] Training Complete!")
    return model, trainer


if __name__ == "__main__":
    print("[BRAIN] Non-Lobotomy Temporal Association Training")
    print("Using polynomial co-prime functionals and evolutionary trust selection")
    print("=" * 75)
    try:
        model, trainer = run_non_lobotomy_temporal_training()
        print(f"\n[OK] Training completed successfully!")
    except Exception as e:
        print(f"\n[ERR] Training failed: {e}")
        import traceback
        traceback.print_exc()
