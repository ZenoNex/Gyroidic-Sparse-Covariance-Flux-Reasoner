"""
Simple Temporal Association Training Script

This is a standalone script that demonstrates temporal association training
without complex import dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os
from typing import Dict, List, Tuple, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import repair components directly
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from src.core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from src.core.love_invariant_protector import LoveInvariantProtector, SoftSaturatedGates
from src.optimization.codes_driver import CODES


class SimpleGyroidModel(nn.Module):
    """
    Simplified Gyroid model for temporal association training.
    
    This is a minimal version that includes the repair system
    but avoids complex dependencies.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_functionals: int = 5,
        poly_degree: int = 12,
        device: str = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_functionals
        self.D = poly_degree + 1
        self.device = device
        
        # Simple embedding layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Residue generation
        self.residue_proj = nn.Linear(hidden_dim, self.K * self.D)
        
        # Trust scalars (learnable parameters)
        self.register_buffer('trust_scalars', torch.ones(self.K, device=device))
        
        # Repair system components
        self.spectral_corrector = SpectralCoherenceCorrector(device=device)
        self.bezout_refresh = BezoutCoefficientRefresh(self.K, poly_degree, device=device)
        self.chern_simons_gasket = ChernSimonsGasket(device=device)
        self.soliton_healer = SolitonStabilityHealer(device=device)
        self.love_protector = LoveInvariantProtector(hidden_dim, device=device)
        self.soft_gates = SoftSaturatedGates(self.K, poly_degree, device=device)
        
        # CODES Driver for proper PAS_h computation
        self.codes_driver = CODES(coherence_threshold=0.75)
        
        # Memory for temporal coherence
        self.register_buffer('prev_state', torch.zeros(1, hidden_dim, device=device))
    
    def forward(self, x: torch.Tensor, return_analysis: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with repair system."""
        batch_size = x.shape[0]
        
        # Input projection
        h = torch.relu(self.input_proj(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        
        # REPAIR PHASE 1: Spectral coherence correction
        h = self.spectral_corrector.adaptive_coherence_correction(h.unsqueeze(1)).squeeze(1)
        
        # Generate residues
        residue_flat = self.residue_proj(h)
        residue_distributions = residue_flat.view(batch_size, self.K, self.D)
        
        # REPAIR PHASE 2: Bezout coefficient refresh
        residue_distributions = self.bezout_refresh.apply_crt_correction(residue_distributions)
        
        # REPAIR PHASE 3: Chern-Simons gasket
        # Use proper polynomial co-prime functional system (anti-lobotomy)
        if not hasattr(self, 'polynomial_config'):
            # Initialize polynomial co-prime system if not exists
            from src.core.polynomial_coprime import PolynomialCoprimeConfig
            self.polynomial_config = PolynomialCoprimeConfig(
                k=self.K,
                degree=self.D - 1,
                basis_type='chebyshev',
                learnable=True,
                use_saturation=True,
                device=self.device
            )
        
        # Get polynomial coefficients from the proper system
        poly_coeffs = self.polynomial_config.get_coefficients_tensor()  # [K, D]
        residue_distributions = self.chern_simons_gasket.plug_logic_leak(
            residue_distributions, poly_coeffs
        )
        
        # REPAIR PHASE 4: Love invariant protection
        love_vector, love_diagnostics = self.love_protector.apply_love_protection(h)
        
        # REPAIR PHASE 5: Soft saturated gates with proper PAS_h
        pas_h = self._compute_pas_h(residue_distributions)
        residue_distributions = self.soft_gates.apply_soft_saturation(residue_distributions, pas_h)
        
        # REPAIR PHASE 6: Soliton healing
        residue_distributions = self.soliton_healer.heal_fractured_soliton(residue_distributions)
        
        # Output projection
        output = self.output_proj(h)
        
        # Update previous state for temporal coherence
        self.prev_state = h.detach().mean(dim=0, keepdim=True)
        
        results = {
            'output': output.squeeze(-1),
            'hidden_state': h,
            'residue_distributions': residue_distributions,
            'trust_scalars': self.trust_scalars.clone()
        }
        
        if return_analysis:
            results.update({
                'spectral_diagnostics': self.spectral_corrector.get_diagnostics(),
                'chern_simons_diagnostics': self.chern_simons_gasket.get_diagnostics(),
                'soliton_healing_diagnostics': self.soliton_healer.get_diagnostics(),
                'love_diagnostics': love_diagnostics,
                'soft_gates_diagnostics': self.soft_gates.get_diagnostics()
            })
        
        return results
    
    def _compute_pas_h(self, residue_distributions: torch.Tensor) -> float:
        """
        Compute Phase Alignment Score using proper CODES multiharmonic alignment.
        
        PAS_h = Σ_{d=0}^D (1/(d+1)) * ||θ_d||_2
        
        This follows the INVARIANT_OPTIMIZATION.md specification.
        """
        if not hasattr(self, 'polynomial_config'):
            # If polynomial config not initialized, use simple fallback
            return 0.5
        
        # Get polynomial coefficients tensor [K, D]
        theta = self.polynomial_config.get_coefficients_tensor()
        
        # Compute multiharmonic phase alignment score
        pas_h = 0.0
        D = theta.shape[1]  # Polynomial degree + 1
        
        for d in range(D):
            # Harmonic weight: 1/(d+1) - higher weight for lower degrees
            harmonic_weight = 1.0 / (d + 1)
            
            # L2 norm of degree-d coefficients across all functionals
            theta_d_norm = torch.norm(theta[:, d]).item()
            
            # Weighted contribution
            pas_h += harmonic_weight * theta_d_norm
        
        # Use CODES driver for additional phase coherence computation
        # Derive phase from residue tensor statistics
        residue_phase = float(torch.sum(residue_distributions).item() % (2 * math.pi))
        codes_coherence = self.codes_driver.compute_pas_h(residue_phase)
        
        # Combine polynomial harmonic alignment with CODES coherence
        combined_pas_h = 0.7 * pas_h + 0.3 * codes_coherence
        
        return combined_pas_h
    
    def compute_temporal_coherence(self, current_state: torch.Tensor) -> torch.Tensor:
        """Compute temporal coherence with previous state."""
        if torch.allclose(self.prev_state, torch.zeros_like(self.prev_state)):
            return torch.tensor(1.0, device=self.device)  # First step
        
        # Coherence = smooth transition
        transition_diff = torch.norm(current_state.mean(dim=0) - self.prev_state.squeeze())
        coherence = 1.0 / (1.0 + transition_diff)
        return coherence


class SimpleTemporalDataset:
    """Simple temporal dataset for association training."""
    
    def __init__(
        self,
        sequence_length: int = 16,
        num_concepts: int = 100,
        embedding_dim: int = 768,
        device: str = None
    ):
        self.sequence_length = sequence_length
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Create concept embeddings
        self.concept_embeddings = torch.randn(num_concepts, embedding_dim, device=device)
        
        # Create simple association patterns
        self.associations = self._create_associations()
    
    def _create_associations(self) -> Dict[int, List[int]]:
        """Create simple concept associations."""
        associations = {}
        
        # Create clusters of 10 concepts each
        cluster_size = 10
        num_clusters = self.num_concepts // cluster_size
        
        for cluster_id in range(num_clusters):
            cluster_start = cluster_id * cluster_size
            cluster_end = cluster_start + cluster_size
            
            for concept_id in range(cluster_start, cluster_end):
                # Associate with other concepts in same cluster
                cluster_concepts = list(range(cluster_start, cluster_end))
                cluster_concepts.remove(concept_id)
                associations[concept_id] = cluster_concepts[:5]  # Top 5 associations
        
        return associations
    
    def get_batch(self, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Generate a batch of temporal sequences."""
        sequences = []
        targets = []
        
        for _ in range(batch_size):
            # Create a sequence
            sequence = []
            sequence_targets = []
            
            # Start with random concept
            current_concept = np.random.randint(self.num_concepts)
            
            for step in range(self.sequence_length):
                # Add current concept embedding
                sequence.append(self.concept_embeddings[current_concept])
                
                # Target is associated concepts
                if current_concept in self.associations:
                    target_concepts = self.associations[current_concept]
                    target_embedding = self.concept_embeddings[target_concepts[0]]  # Use first association
                else:
                    target_embedding = self.concept_embeddings[current_concept]  # Self-association
                
                sequence_targets.append(target_embedding)
                
                # Move to next concept (with some randomness)
                if current_concept in self.associations and np.random.random() > 0.3:
                    current_concept = np.random.choice(self.associations[current_concept])
                else:
                    current_concept = np.random.randint(self.num_concepts)
            
            sequences.append(torch.stack(sequence))
            targets.append(torch.stack(sequence_targets))
        
        return {
            'sequences': torch.stack(sequences),  # [batch, seq_len, embed_dim]
            'targets': torch.stack(targets)       # [batch, seq_len, embed_dim]
        }


class SimpleTemporalTrainer:
    """Simple trainer for temporal associations."""
    
    def __init__(
        self,
        model: SimpleGyroidModel,
        dataset: SimpleTemporalDataset,
        learning_rate: float = 1e-4,
        trust_update_rate: float = 0.02,
        fossilization_threshold: float = 0.85
    ):
        self.model = model
        self.dataset = dataset
        self.trust_update_rate = trust_update_rate
        self.fossilization_threshold = fossilization_threshold
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'association_accuracy': [],
            'temporal_coherence': [],
            'trust_evolution': [],
            'fossilization_events': []
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        sequences = batch['sequences']  # [batch, seq_len, embed_dim]
        targets = batch['targets']      # [batch, seq_len, embed_dim]
        
        batch_size, seq_len, embed_dim = sequences.shape
        
        total_loss = 0.0
        coherence_scores = []
        
        # Process sequence step by step
        for t in range(seq_len):
            # Current input
            current_input = sequences[:, t, :]  # [batch, embed_dim]
            current_target = targets[:, t, :]   # [batch, embed_dim]
            
            # Forward pass
            output = self.model(current_input, return_analysis=True)
            
            # Association loss (cosine similarity)
            hidden_state = output['hidden_state']
            
            # Project target to hidden dimension for comparison
            # Use a simple linear projection to match dimensions
            if not hasattr(self, 'target_projector'):
                self.target_projector = nn.Linear(768, 256).to(hidden_state.device)
            
            target_proj = self.target_projector(current_target)
            association_loss = 1.0 - torch.cosine_similarity(hidden_state, target_proj, dim=1).mean()
            
            # Temporal coherence
            coherence = self.model.compute_temporal_coherence(hidden_state)
            coherence_scores.append(coherence)
            
            # Total survivorship pressure
            step_loss = association_loss - 0.1 * coherence
            total_loss += step_loss
        
        # Average over sequence
        avg_loss = total_loss / seq_len
        avg_coherence = torch.stack(coherence_scores).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update trust scalars
        association_accuracy = 1.0 - association_loss.item()
        self.update_trust_scalars(association_accuracy)
        
        return {
            'survivorship_pressure': avg_loss.item(),
            'association_accuracy': association_accuracy,
            'temporal_coherence': avg_coherence.item(),
            'trust_mean': self.model.trust_scalars.mean().item(),
            'trust_std': self.model.trust_scalars.std().item(),
            'num_fossilized': (self.model.trust_scalars > self.fossilization_threshold).sum().item()
        }
    
    def update_trust_scalars(self, association_accuracy: float):
        """Update trust scalars based on performance."""
        current_trust = self.model.trust_scalars.clone()
        
        # Simple trust update
        trust_delta = self.trust_update_rate * (association_accuracy - 0.5)
        new_trust = torch.clamp(current_trust + trust_delta, 0.0, 1.0)
        
        self.model.trust_scalars.copy_(new_trust)
        
        # Check for fossilization events
        newly_fossilized = (new_trust > self.fossilization_threshold) & (current_trust <= self.fossilization_threshold)
        if newly_fossilized.any():
            fossilized_indices = torch.where(newly_fossilized)[0]
            self.history['fossilization_events'].append({
                'step': len(self.history['association_accuracy']),
                'fossilized_functionals': fossilized_indices.tolist()
            })
            print(f"🔒 Fossilized functionals: {fossilized_indices.tolist()}")
    
    def train_epoch(self, num_batches: int = 50) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = []
        
        for batch_idx in range(num_batches):
            # Generate batch
            batch = self.dataset.get_batch(batch_size=4)
            
            # Train step
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:2d}: "
                      f"Assoc={metrics['association_accuracy']:.3f}, "
                      f"Coherence={metrics['temporal_coherence']:.3f}, "
                      f"Trust={metrics['trust_mean']:.3f}±{metrics['trust_std']:.3f}")
        
        # Compute epoch averages
        epoch_summary = {}
        for key in ['survivorship_pressure', 'association_accuracy', 'temporal_coherence', 'trust_mean', 'trust_std']:
            epoch_summary[key] = np.mean([m[key] for m in epoch_metrics])
        
        epoch_summary['final_num_fossilized'] = epoch_metrics[-1]['num_fossilized']
        
        # Update history
        self.history['association_accuracy'].append(epoch_summary['association_accuracy'])
        self.history['temporal_coherence'].append(epoch_summary['temporal_coherence'])
        self.history['trust_evolution'].append([m['trust_mean'] for m in epoch_metrics])
        
        return epoch_summary


def run_simple_temporal_training():
    """Run simple temporal association training."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Simple Temporal Association Training")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Create model
    print("🏗️ Creating model...")
    model = SimpleGyroidModel(
        input_dim=768,
        hidden_dim=256,
        num_functionals=5,
        poly_degree=4,
        device=device
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Trust scalars: {model.trust_scalars.tolist()}")
    
    # Create dataset
    print("📊 Creating dataset...")
    dataset = SimpleTemporalDataset(
        sequence_length=8,
        num_concepts=50,
        embedding_dim=768,
        device=device
    )
    
    print(f"✅ Dataset created with {dataset.num_concepts} concepts")
    print(f"   Association example: {list(dataset.associations.items())[:2]}")
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = SimpleTemporalTrainer(
        model=model,
        dataset=dataset,
        learning_rate=1e-4,
        trust_update_rate=0.02,
        fossilization_threshold=0.85
    )
    
    print("✅ Trainer created")
    
    # Test with sample batch
    print("\n🧪 Testing with sample batch...")
    sample_batch = dataset.get_batch(batch_size=2)
    print(f"   Sequences shape: {sample_batch['sequences'].shape}")
    print(f"   Targets shape: {sample_batch['targets'].shape}")
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(sample_batch['sequences'][0, 0, :].unsqueeze(0), return_analysis=True)
        print(f"   Test output shape: {test_output['output'].shape}")
        print(f"   Repair diagnostics: {list(test_output.keys())}")
    
    # Training loop
    num_epochs = 5
    batches_per_epoch = 30
    
    print(f"\n🚀 Starting training: {num_epochs} epochs, {batches_per_epoch} batches each")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        
        # Train epoch
        epoch_metrics = trainer.train_epoch(num_batches=batches_per_epoch)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Association Accuracy: {epoch_metrics['association_accuracy']:.3f}")
        print(f"   Temporal Coherence: {epoch_metrics['temporal_coherence']:.3f}")
        print(f"   Trust Mean: {epoch_metrics['trust_mean']:.3f} ± {epoch_metrics['trust_std']:.3f}")
        print(f"   Fossilized Functionals: {epoch_metrics['final_num_fossilized']}")
        
        # Show current trust scalars
        current_trust = model.trust_scalars
        print(f"   Trust Scalars: {[f'{t:.3f}' for t in current_trust.tolist()]}")
        
        # Test repair system
        print(f"\n🔧 Repair System Status:")
        with torch.no_grad():
            test_input = torch.randn(1, 768, device=device)
            repair_output = model(test_input, return_analysis=True)
            
            spectral_diag = repair_output.get('spectral_diagnostics', {})
            print(f"   Spectral coherence threshold: {spectral_diag.get('theta_coherence', 'N/A')}")
            
            love_diag = repair_output.get('love_diagnostics', {})
            print(f"   Love violations: {love_diag.get('violation_count', 'N/A')}")
            
            soft_gates_diag = repair_output.get('soft_gates_diagnostics', {})
            print(f"   System temperature: {soft_gates_diag.get('dt', 'N/A')}")
    
    # Final results
    print(f"\n🎯 Training Complete!")
    print("=" * 50)
    
    final_trust = model.trust_scalars
    print(f"Final Trust Scalars: {[f'{t:.3f}' for t in final_trust.tolist()]}")
    print(f"Trust Evolution: {trainer.history['association_accuracy']}")
    print(f"Fossilization Events: {len(trainer.history['fossilization_events'])}")
    
    # Test final system
    print(f"\n🧪 Final System Test:")
    test_cases = [
        "High variance input (chaos)",
        "Low variance input (order)",
        "Random input (baseline)"
    ]
    
    for i, test_case in enumerate(test_cases):
        if i == 0:
            test_input = torch.randn(1, 768, device=device) * 2.0  # High variance
        elif i == 1:
            test_input = torch.ones(1, 768, device=device) * 0.1   # Low variance
        else:
            test_input = torch.randn(1, 768, device=device)        # Normal
        
        with torch.no_grad():
            output = model(test_input, return_analysis=True)
            print(f"   {test_case}: Output norm = {torch.norm(output['output']):.3f}")
    
    return model, trainer


if __name__ == "__main__":
    print("🧠 Simple Temporal Association Training")
    print("This script demonstrates temporal association training with repair system")
    print("=" * 70)
    
    try:
        model, trainer = run_simple_temporal_training()
        
        print(f"\n✅ Training completed successfully!")
        print(f"The system now has:")
        print(f"• Temporal association memory")
        print(f"• Evolved trust scalars")
        print(f"• Active repair system")
        print(f"• Fossilization capability")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
