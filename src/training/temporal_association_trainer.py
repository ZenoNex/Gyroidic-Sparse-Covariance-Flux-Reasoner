"""
Temporal Association Trainer: Feeds the repaired system with ordered data and associations.

This trainer provides the Gyroidic Flux Reasoner with temporal sequences and 
associative patterns to build up its resonance cavity memory and trust scalars.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json
import time

from src.models.gyroid_reasoner import GyroidicFluxReasoner
from src.models.resonance_cavity import ResonanceCavity

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))



class TemporalAssociationDataset:
    """
    Dataset that provides temporal sequences and associative patterns.
    
    This feeds the system with:
    1. Sequential patterns (A → B → C)
    2. Associative clusters (related concepts)
    3. Temporal dependencies (cause → effect)
    4. Contextual embeddings (meaning in context)
    """
    
    def __init__(
        self,
        sequence_length: int = 32,
        association_window: int = 8,
        num_concepts: int = 1000,
        device: str = None
    ):
        self.sequence_length = sequence_length
        self.association_window = association_window
        self.num_concepts = num_concepts
        self.device = device
        
        # Generate concept embeddings
        self.concept_embeddings = torch.randn(num_concepts, 768, device=device)
        
        # Create association graph (concepts that appear together)
        self.association_graph = self._create_association_graph()
        
        # Temporal patterns (sequences that commonly occur)
        self.temporal_patterns = self._create_temporal_patterns()
        
        # Contextual modifiers (how context changes meaning)
        self.contextual_modifiers = self._create_contextual_modifiers()
    
    def _create_association_graph(self) -> Dict[int, List[int]]:
        """Create graph of concept associations."""
        graph = {}
        
        # Create clusters of related concepts
        cluster_size = 20
        num_clusters = self.num_concepts // cluster_size
        
        for cluster_id in range(num_clusters):
            cluster_concepts = list(range(
                cluster_id * cluster_size, 
                (cluster_id + 1) * cluster_size
            ))
            
            # Each concept in cluster is associated with others
            for concept in cluster_concepts:
                # Strong associations within cluster
                graph[concept] = [c for c in cluster_concepts if c != concept]
                
                # Weak associations with other clusters
                if cluster_id > 0:
                    prev_cluster_concept = (cluster_id - 1) * cluster_size + np.random.randint(cluster_size)
                    graph[concept].append(prev_cluster_concept)
                
                if cluster_id < num_clusters - 1:
                    next_cluster_concept = (cluster_id + 1) * cluster_size + np.random.randint(cluster_size)
                    graph[concept].append(next_cluster_concept)
        
        return graph
    
    def _create_temporal_patterns(self) -> List[List[int]]:
        """Create common temporal sequences."""
        patterns = []
        
        # Create various types of patterns
        for _ in range(100):
            pattern_type = np.random.choice(['linear', 'branching', 'cyclic'])
            
            if pattern_type == 'linear':
                # A → B → C → D
                start_concept = np.random.randint(self.num_concepts)
                pattern = [start_concept]
                for _ in range(np.random.randint(3, 8)):
                    if pattern[-1] in self.association_graph:
                        next_concept = np.random.choice(self.association_graph[pattern[-1]])
                        pattern.append(next_concept)
                patterns.append(pattern)
            
            elif pattern_type == 'branching':
                # A → B → C, A → B → D
                root = np.random.randint(self.num_concepts)
                branch_point = np.random.choice(self.association_graph.get(root, [root]))
                
                for _ in range(2):  # Two branches
                    pattern = [root, branch_point]
                    if branch_point in self.association_graph:
                        branch_end = np.random.choice(self.association_graph[branch_point])
                        pattern.append(branch_end)
                    patterns.append(pattern)
            
            elif pattern_type == 'cyclic':
                # A → B → C → A
                start_concept = np.random.randint(self.num_concepts)
                pattern = [start_concept]
                current = start_concept
                
                for _ in range(np.random.randint(2, 5)):
                    if current in self.association_graph:
                        current = np.random.choice(self.association_graph[current])
                        pattern.append(current)
                
                pattern.append(start_concept)  # Close the cycle
                patterns.append(pattern)
        
        return patterns
    
    def _create_contextual_modifiers(self) -> Dict[str, torch.Tensor]:
        """Create context-dependent meaning modifiers."""
        contexts = ['positive', 'negative', 'temporal', 'spatial', 'causal']
        modifiers = {}
        
        for context in contexts:
            # Each context has a transformation matrix
            modifiers[context] = torch.randn(768, 768, device=self.device) * 0.1
        
        return modifiers
    
    def get_temporal_sequence(self, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Generate a batch of temporal sequences."""
        sequences = []
        associations = []
        contexts = []
        
        for _ in range(batch_size):
            # Choose a temporal pattern
            pattern = np.random.choice(self.temporal_patterns)
            
            # Extend or truncate to sequence_length
            if len(pattern) < self.sequence_length:
                # Extend with associations
                while len(pattern) < self.sequence_length:
                    last_concept = pattern[-1]
                    if last_concept in self.association_graph:
                        next_concept = np.random.choice(self.association_graph[last_concept])
                        pattern.append(next_concept)
                    else:
                        pattern.append(np.random.randint(self.num_concepts))
            else:
                pattern = pattern[:self.sequence_length]
            
            # Get embeddings for sequence
            sequence_embeddings = self.concept_embeddings[pattern]
            
            # Apply contextual modification
            context_type = np.random.choice(list(self.contextual_modifiers.keys()))
            context_modifier = self.contextual_modifiers[context_type]
            modified_embeddings = torch.matmul(sequence_embeddings, context_modifier)
            
            sequences.append(modified_embeddings)
            
            # Create association targets (concepts that should be associated)
            association_targets = []
            for concept in pattern:
                if concept in self.association_graph:
                    targets = self.association_graph[concept][:self.association_window]
                    while len(targets) < self.association_window:
                        targets.append(concept)  # Self-association
                    association_targets.extend(targets)
                else:
                    association_targets.extend([concept] * self.association_window)
            
            associations.append(torch.tensor(association_targets[:self.sequence_length * self.association_window]))
            contexts.append(context_type)
        
        return {
            'sequences': torch.stack(sequences),  # [batch, seq_len, 768]
            'associations': torch.stack(associations),  # [batch, seq_len * assoc_window]
            'contexts': contexts,  # List of context types
            'concept_ids': torch.tensor([pattern for pattern in [np.random.choice(self.temporal_patterns) for _ in range(batch_size)]])
        }


class TemporalAssociationTrainer:
    """
    Trainer that feeds temporal associations to the Gyroidic Flux Reasoner.
    
    This trainer:
    1. Provides sequential data to build up resonance cavity memory
    2. Trains trust scalars through successful associations
    3. Builds up fossilized patterns through repetition
    4. Creates temporal dependencies in the system
    """
    
    def __init__(
        self,
        model: GyroidicFluxReasoner,
        dataset: TemporalAssociationDataset,
        learning_rate: float = 1e-4,
        trust_update_rate: float = 0.01,
        fossilization_threshold: float = 0.8,
        device: str = None
    ):
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.trust_update_rate = trust_update_rate
        self.fossilization_threshold = fossilization_threshold
        self.device = device
        
        # Optimizer for model parameters (excluding fossilized ones)
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], 
            lr=learning_rate
        )
        
        # Training history
        self.training_history = {
            'trust_evolution': [],
            'fossilization_events': [],
            'association_accuracy': [],
            'temporal_coherence': [],
            'repair_diagnostics': []
        }

    def _call_model(
        self,
        text_emb: 'torch.Tensor',
        return_analysis: bool = True,
    ) -> dict:
        """
        Dispatch helper so the trainer works both with:
          - DiegeticPhysicsEngine  (has forward_text_emb adapter)
          - Original GyroidicFluxReasoner  (responds to __call__(text_emb=...))
        """
        if hasattr(self.model, 'forward_text_emb'):
            return self.model.forward_text_emb(
                text_emb, return_analysis=return_analysis
            )
        # Legacy fallback — original model interface
        return self.model(text_emb=text_emb, return_analysis=return_analysis)

    def compute_association_loss(
        self, 
        model_output: Dict[str, torch.Tensor], 
        target_associations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss based on association accuracy.
        
        This is NOT a teleological loss - it's a survivorship pressure.
        The system survives if it can maintain associations.
        """
        # Get residue distributions from model
        residue_distributions = model_output.get('residue_distributions')
        if residue_distributions is None:
            return torch.tensor(0.0, device=self.device)
        
        batch_size, K, D = residue_distributions.shape
        
        # Flatten residues to compare with associations
        flat_residues = residue_distributions.view(batch_size, -1)
        
        # Target associations (flattened)
        target_flat = target_associations.float()
        
        # Ensure same size
        min_size = min(flat_residues.shape[1], target_flat.shape[1])
        flat_residues = flat_residues[:, :min_size]
        target_flat = target_flat[:, :min_size]
        
        # Association pressure: how well residues correlate with expected associations
        correlation_loss = 1.0 - torch.cosine_similarity(flat_residues, target_flat, dim=1).mean()
        
        return correlation_loss
    
    def compute_temporal_coherence(
        self, 
        sequence_outputs: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute temporal coherence across sequence.
        
        Measures how well the system maintains coherent state transitions.
        """
        if len(sequence_outputs) < 2:
            return torch.tensor(0.0, device=self.device)
        
        coherence_scores = []
        
        for i in range(len(sequence_outputs) - 1):
            current_state = sequence_outputs[i].get('reconstruction', torch.zeros(1, device=self.device))
            next_state = sequence_outputs[i + 1].get('reconstruction', torch.zeros(1, device=self.device))
            
            # Coherence = smooth transition (not too abrupt)
            transition_smoothness = 1.0 / (1.0 + torch.norm(next_state - current_state))
            coherence_scores.append(transition_smoothness)
        
        return torch.stack(coherence_scores).mean()
    
    def update_trust_scalars(
        self, 
        model_output: Dict[str, torch.Tensor], 
        association_accuracy: float
    ):
        """
        Update trust scalars based on association performance.
        
        Successful functionals get higher trust, unsuccessful ones get lower trust.
        """
        # Get current trust scalars
        current_trust = self.model.trust_scalars.clone()
        
        # Get functional performance from model diagnostics
        if 'crt_pressure' in model_output:
            crt_pressure = model_output['crt_pressure'].value
            
            # Lower pressure = better performance = higher trust
            performance_bonus = torch.exp(-crt_pressure)
            
            # Update trust with association accuracy
            trust_update = self.trust_update_rate * (association_accuracy - 0.5) * performance_bonus
            
            # Apply update
            new_trust = torch.clamp(current_trust + trust_update, 0.0, 1.0)
            self.model.trust_scalars.copy_(new_trust)
            
            # Check for fossilization
            fossilized_mask = new_trust > self.fossilization_threshold
            if fossilized_mask.any():
                fossilized_indices = torch.where(fossilized_mask)[0]
                self.training_history['fossilization_events'].append({
                    'step': len(self.training_history['trust_evolution']),
                    'fossilized_functionals': fossilized_indices.tolist(),
                    'trust_values': new_trust[fossilized_indices].tolist()
                })
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with temporal associations."""
        
        sequences = batch_data['sequences']  # [batch, seq_len, 768]
        associations = batch_data['associations']  # [batch, seq_len * assoc_window]
        contexts = batch_data['contexts']
        
        batch_size, seq_len, embed_dim = sequences.shape
        
        # Process sequence step by step to build temporal dependencies
        sequence_outputs = []
        total_association_loss = 0.0
        
        for t in range(seq_len):
            # Get current step input
            current_input = sequences[:, t, :]  # [batch, 768]
            
            # Run model via adapter
            model_output = self._call_model(
                text_emb=current_input,
                return_analysis=True,
            )
            
            sequence_outputs.append(model_output)
            
            # Compute association loss for this step
            step_associations = associations[:, t * self.dataset.association_window:(t + 1) * self.dataset.association_window]
            association_loss = self.compute_association_loss(model_output, step_associations)
            total_association_loss += association_loss
        
        # Average association loss
        avg_association_loss = total_association_loss / seq_len
        
        # Compute temporal coherence
        temporal_coherence = self.compute_temporal_coherence(sequence_outputs)
        
        # Total survivorship pressure (not teleological loss)
        survivorship_pressure = avg_association_loss - 0.1 * temporal_coherence
        
        # Backward pass
        self.optimizer.zero_grad()
        survivorship_pressure.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update trust scalars
        association_accuracy = 1.0 - avg_association_loss.item()
        self.update_trust_scalars(sequence_outputs[-1], association_accuracy)
        
        # Collect diagnostics
        final_output = sequence_outputs[-1]
        repair_diagnostics = {
            'spectral': final_output.get('spectral_diagnostics', {}),
            'chern_simons': final_output.get('chern_simons_diagnostics', {}),
            'soliton_healing': final_output.get('soliton_healing_diagnostics', {}),
            'love': final_output.get('love_diagnostics', {}),
            'soft_gates': final_output.get('soft_gates_diagnostics', {})
        }
        
        return {
            'survivorship_pressure': survivorship_pressure.item(),
            'association_accuracy': association_accuracy,
            'temporal_coherence': temporal_coherence.item(),
            'trust_mean': self.model.trust_scalars.mean().item(),
            'trust_std': self.model.trust_scalars.std().item(),
            'num_fossilized': (self.model.trust_scalars > self.fossilization_threshold).sum().item(),
            'repair_diagnostics': repair_diagnostics
        }
    
    def train_epoch(self, num_batches: int = 100) -> Dict[str, float]:
        """Train for one epoch."""
        
        epoch_metrics = {
            'survivorship_pressure': [],
            'association_accuracy': [],
            'temporal_coherence': [],
            'trust_evolution': [],
            'repair_metrics': []
        }
        
        for batch_idx in range(num_batches):
            # Generate batch
            batch_data = self.dataset.get_temporal_sequence(batch_size=4)
            
            # Train step
            step_metrics = self.train_step(batch_data)
            
            # Collect metrics
            for key in ['survivorship_pressure', 'association_accuracy', 'temporal_coherence']:
                epoch_metrics[key].append(step_metrics[key])
            
            epoch_metrics['trust_evolution'].append({
                'mean': step_metrics['trust_mean'],
                'std': step_metrics['trust_std'],
                'num_fossilized': step_metrics['num_fossilized']
            })
            
            epoch_metrics['repair_metrics'].append(step_metrics['repair_diagnostics'])
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx:3d}: "
                      f"Assoc={step_metrics['association_accuracy']:.3f}, "
                      f"Coherence={step_metrics['temporal_coherence']:.3f}, "
                      f"Trust={step_metrics['trust_mean']:.3f}±{step_metrics['trust_std']:.3f}, "
                      f"Fossilized={step_metrics['num_fossilized']}")
        
        # Compute epoch averages
        epoch_summary = {}
        for key in ['survivorship_pressure', 'association_accuracy', 'temporal_coherence']:
            epoch_summary[key] = np.mean(epoch_metrics[key])
        
        epoch_summary['final_trust_mean'] = epoch_metrics['trust_evolution'][-1]['mean']
        epoch_summary['final_num_fossilized'] = epoch_metrics['trust_evolution'][-1]['num_fossilized']
        
        # Update training history
        self.training_history['association_accuracy'].append(epoch_summary['association_accuracy'])
        self.training_history['temporal_coherence'].append(epoch_summary['temporal_coherence'])
        self.training_history['trust_evolution'].append(epoch_metrics['trust_evolution'])
        self.training_history['repair_diagnostics'].append(epoch_metrics['repair_metrics'])
        
        return epoch_summary
    
    def save_training_state(self, filepath: str):
        """Save training state and history."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'trust_scalars': self.model.trust_scalars.clone(),
            'dataset_config': {
                'num_concepts': self.dataset.num_concepts,
                'sequence_length': self.dataset.sequence_length,
                'association_window': self.dataset.association_window
            }
        }
        
        torch.save(state, filepath)
        print(f"Training state saved to {filepath}")
    
    def load_training_state(self, filepath: str):
        """Load training state and history."""
        state = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.training_history = state['training_history']
        self.model.trust_scalars.copy_(state['trust_scalars'])
        
        print(f"Training state loaded from {filepath}")
        print(f"Resumed with {state['trust_scalars'].sum().item():.2f} total trust, "
              f"{(state['trust_scalars'] > self.fossilization_threshold).sum().item()} fossilized functionals")


def create_training_session(
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    training_config: Dict[str, Any],
    device: str = None
) -> TemporalAssociationTrainer:
    """Create a complete training session."""
    
    # Create model
    model = GyroidicFluxReasoner(**model_config).to(device, non_blocking=True)
    
    # Create dataset
    dataset = TemporalAssociationDataset(**dataset_config, device=device)
    
    # Create trainer
    trainer = TemporalAssociationTrainer(
        model=model,
        dataset=dataset,
        **training_config,
        device=device
    )
    
    return trainer

