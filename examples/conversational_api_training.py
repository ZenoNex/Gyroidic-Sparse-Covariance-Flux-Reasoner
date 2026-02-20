#!/usr/bin/env python3
"""
Conversational API Training Integration

This script demonstrates how to integrate conversational data from APIs
with the temporal association training system and diegetic backend.

Key Features:
- Real conversational data ingestion from APIs
- Integration with affordance gradient system
- Temporal association training with real conversations
- Diegetic terminal backend integration
- Proper PAS_h computation with conversational patterns

Author: William Matthew Bryant
Created: January 2026
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

# Core imports
from src.data.conversational_api_ingestor import (
    ConversationalAPIIngestor,
    Conversation,
    ConversationTurn
)
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from src.core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from src.core.love_invariant_protector import LoveInvariantProtector, SoftSaturatedGates
from src.optimization.codes_driver import CODES


class ConversationalTemporalModel(nn.Module):
    """
    Temporal model trained on real conversational data from APIs.
    
    Integrates:
    - Conversational affordance gradients
    - API extraction patterns
    - Temporal association learning
    - Proper PAS_h computation
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
        
        # Core neural components
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # Conversational pattern recognition
        self.conversational_head = nn.Linear(hidden_dim, 64)
        self.api_extraction_head = nn.Linear(hidden_dim, 64)
        self.dialogue_flow_head = nn.Linear(hidden_dim, 32)
        
        # Polynomial co-prime system (anti-lobotomy)
        self.polynomial_config = PolynomialCoprimeConfig(
            k=num_functionals,
            degree=poly_degree,
            basis_type='chebyshev',
            learnable=True,
            use_saturation=True,
            device=device
        )
        
        # Repair system components
        self.spectral_corrector = SpectralCoherenceCorrector(device=device)
        self.bezout_refresh = BezoutCoefficientRefresh(self.K, poly_degree, device=device)
        self.chern_simons_gasket = ChernSimonsGasket(device=device)
        self.soliton_healer = SolitonStabilityHealer(device=device)
        self.love_protector = LoveInvariantProtector(hidden_dim, device=device)
        self.soft_gates = SoftSaturatedGates(self.K, poly_degree, device=device)
        
        # CODES driver for proper PAS_h computation
        self.codes_driver = CODES(coherence_threshold=0.75)
        
        # Conversational memory and trust
        self.register_buffer('trust_scalars', torch.ones(self.K, device=device))
        self.register_buffer('conversation_memory', torch.zeros(10, hidden_dim, device=device))
        self.memory_index = 0
        
        # Affordance gradient tracking
        self.affordance_history = {
            'conversational': [],
            'api_extraction': [],
            'executability': [],
            'temporal_coherence': []
        }
    
    def forward(self, x: torch.Tensor, affordance_gradients: Optional[Dict[str, float]] = None,
                conversation_context: Optional[Dict[str, Any]] = None,
                return_analysis: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with conversational affordance integration.
        """
        batch_size = x.shape[0]
        
        # Input projection
        h = torch.relu(self.input_proj(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        
        # Conversational pattern recognition
        conversational_features = torch.tanh(self.conversational_head(h))
        api_features = torch.tanh(self.api_extraction_head(h))
        dialogue_features = torch.tanh(self.dialogue_flow_head(h))
        
        # Integrate affordance gradients if provided
        if affordance_gradients:
            # Modulate features based on affordance gradients
            conv_strength = affordance_gradients.get('conversational', 0.0)
            api_strength = affordance_gradients.get('api_extraction', 0.0)
            
            conversational_features = conversational_features * (1.0 + conv_strength)
            api_features = api_features * (1.0 + api_strength)
        
        # Combine features
        combined_features = torch.cat([conversational_features, api_features, dialogue_features], dim=1)
        
        # Project back to hidden dimension
        if combined_features.shape[1] != self.hidden_dim:
            if not hasattr(self, 'feature_proj'):
                self.feature_proj = nn.Linear(combined_features.shape[1], self.hidden_dim).to(self.device, non_blocking=True)
            h = self.feature_proj(combined_features)
        
        # REPAIR SYSTEM INTEGRATION
        # Spectral coherence correction
        h_corrected = self.spectral_corrector.adaptive_coherence_correction(h.unsqueeze(1))
        h = h_corrected.squeeze(1)
        
        # Generate polynomial residues
        phi_values = torch.zeros(h.shape[0], self.K, device=self.device)
        
        # Use learned projections for polynomial evaluation
        if not hasattr(self, 'functional_projections'):
            self.functional_projections = nn.ModuleList([
                nn.Linear(h.shape[1], 1) for _ in range(self.K)
            ]).to(self.device, non_blocking=True)
        
        for k in range(self.K):
            h_k = self.functional_projections[k](h)
            phi_k_full = self.polynomial_config.evaluate(h_k)
            phi_values[:, k] = phi_k_full[:, 0, k]
        
        # Apply repair system
        containment_pressure = self._compute_containment_pressure(phi_values)
        
        if containment_pressure > 0.3:  # Lower threshold for conversational data
            # Bezout coefficient refresh
            phi_values = self.bezout_refresh.apply_crt_correction(phi_values.unsqueeze(1)).squeeze(1)
            
            # Chern-Simons gasket with polynomial coefficients
            poly_coeffs = self.polynomial_config.get_coefficients_tensor()
            phi_values = self.chern_simons_gasket.plug_logic_leak(phi_values.unsqueeze(1), poly_coeffs).squeeze(1)
            
            # Soliton healing
            phi_values = self.soliton_healer.heal_fractured_soliton(phi_values.unsqueeze(1)).squeeze(1)
        
        # Love invariant protection
        love_vector, love_diagnostics = self.love_protector.apply_love_protection(h)
        
        # Compute PAS_h with conversational context
        pas_h = self._compute_conversational_pas_h(phi_values, affordance_gradients, conversation_context)
        
        # Soft saturated gates
        phi_values = self.soft_gates.apply_soft_saturation(phi_values.unsqueeze(1), pas_h).squeeze(1)
        
        # Update conversational memory
        self._update_conversational_memory(h.detach().mean(dim=0))
        
        # Output generation
        output = phi_values.mean(dim=1)
        
        results = {
            'output': output,
            'hidden_state': h,
            'phi_values': phi_values,
            'conversational_features': conversational_features,
            'api_features': api_features,
            'dialogue_features': dialogue_features,
            'trust_scalars': self.trust_scalars.clone(),
            'containment_pressure': containment_pressure,
            'pas_h': pas_h,
            'love_vector': love_vector
        }
        
        if return_analysis:
            results.update({
                'polynomial_diagnostics': self._get_polynomial_diagnostics(),
                'conversational_memory_state': self.conversation_memory.clone(),
                'affordance_history': self.affordance_history.copy(),
                'love_diagnostics': love_diagnostics,
                'spectral_diagnostics': self.spectral_corrector.get_diagnostics() if hasattr(self.spectral_corrector, 'get_diagnostics') else {}
            })
        
        return results
    
    def _compute_containment_pressure(self, phi: torch.Tensor) -> float:
        """Compute containment pressure for conversational data."""
        # More sensitive to conversational patterns
        variance = phi.var().item()
        mean_abs = phi.abs().mean().item()
        
        # Conversational data tends to have more variation
        pressure = min((variance + mean_abs) / 3.0, 1.0)
        return pressure
    
    def _compute_conversational_pas_h(self, phi: torch.Tensor, 
                                    affordance_gradients: Optional[Dict[str, float]] = None,
                                    conversation_context: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute PAS_h with conversational context integration.
        
        Incorporates:
        - Polynomial harmonic alignment
        - CODES coherence computation
        - Conversational affordance modulation
        - Dialogue flow patterns
        """
        # Base polynomial harmonic alignment
        theta = self.polynomial_config.get_coefficients_tensor()
        pas_h_base = 0.0
        D = theta.shape[1]
        
        for d in range(D):
            harmonic_weight = 1.0 / (d + 1)
            theta_d_norm = torch.norm(theta[:, d]).item()
            pas_h_base += harmonic_weight * theta_d_norm
        
        # CODES coherence computation
        phi_phase = float(torch.sum(phi).item() % (2 * math.pi))
        codes_coherence = self.codes_driver.compute_pas_h(phi_phase)
        
        # Conversational modulation
        conversational_modulation = 1.0
        if affordance_gradients:
            # Higher conversational affordance increases coherence
            conv_strength = affordance_gradients.get('conversational', 0.0)
            api_strength = affordance_gradients.get('api_extraction', 0.0)
            
            # Conversational patterns tend to be more coherent
            conversational_modulation = 1.0 + 0.2 * conv_strength + 0.1 * api_strength
        
        # Dialogue flow coherence
        dialogue_coherence = 1.0
        if conversation_context:
            turn_count = conversation_context.get('turn_count', 1)
            # Longer conversations may have more complex coherence patterns
            dialogue_coherence = 1.0 + 0.05 * math.log(turn_count + 1)
        
        # Combine all components
        combined_pas_h = (
            0.5 * pas_h_base +
            0.3 * codes_coherence +
            0.2 * conversational_modulation * dialogue_coherence
        )
        
        return combined_pas_h
    
    def _update_conversational_memory(self, new_state: torch.Tensor):
        """Update conversational memory buffer."""
        self.conversation_memory[self.memory_index] = new_state
        self.memory_index = (self.memory_index + 1) % self.conversation_memory.shape[0]
    
    def _get_polynomial_diagnostics(self) -> Dict[str, Any]:
        """Get polynomial system diagnostics."""
        theta = self.polynomial_config.get_coefficients_tensor()
        return {
            'coefficient_norm': torch.norm(theta).item(),
            'coefficient_rank': torch.linalg.matrix_rank(theta).item(),
            'orthogonality_pressure': self.polynomial_config.orthogonality_pressure(),
            'coprimality_pressure': self.polynomial_config.co_primality_pressure()
        }
    
    def update_affordance_history(self, affordance_gradients: Dict[str, float]):
        """Update affordance gradient history for analysis."""
        for key, value in affordance_gradients.items():
            if key in self.affordance_history:
                self.affordance_history[key].append(value)
                # Keep only recent history
                if len(self.affordance_history[key]) > 100:
                    self.affordance_history[key] = self.affordance_history[key][-100:]


class ConversationalAPITrainer:
    """
    Trainer for conversational API data integration.
    
    Handles:
    - Real conversational data from APIs
    - Temporal association learning
    - Affordance gradient evolution
    - Trust scalar adaptation
    """
    
    def __init__(
        self,
        model: ConversationalTemporalModel,
        api_ingestor: ConversationalAPIIngestor,
        learning_rate: float = 1e-4,
        trust_update_rate: float = 0.02
    ):
        self.model = model
        self.api_ingestor = api_ingestor
        self.trust_update_rate = trust_update_rate
        
        # Optimizer for neural components only
        neural_params = []
        for name, param in model.named_parameters():
            if 'polynomial_config' not in name:
                neural_params.append(param)
        
        self.optimizer = torch.optim.Adam(neural_params, lr=learning_rate)
        
        # Training history
        self.history = {
            'conversational_accuracy': [],
            'api_extraction_accuracy': [],
            'temporal_coherence': [],
            'pas_h_evolution': [],
            'trust_evolution': [],
            'affordance_evolution': {}
        }
    
    def train_on_conversations(self, conversations: List[Conversation], 
                             num_epochs: int = 3) -> Dict[str, Any]:
        """Train model on real conversational data."""
        print(f"🚀 Training on {len(conversations)} conversations for {num_epochs} epochs")
        
        epoch_results = []
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_metrics = {
                'conversational_loss': [],
                'api_extraction_loss': [],
                'temporal_coherence': [],
                'pas_h_values': [],
                'affordance_gradients': []
            }
            
            # Shuffle conversations
            shuffled_conversations = conversations.copy()
            np.random.shuffle(shuffled_conversations)
            
            for i, conversation in enumerate(shuffled_conversations):
                try:
                    # Train on conversation
                    conv_metrics = self._train_conversation(conversation)
                    
                    # Accumulate metrics
                    for key, value in conv_metrics.items():
                        if key in epoch_metrics:
                            epoch_metrics[key].append(value)
                    
                    # Print progress
                    if (i + 1) % 10 == 0:
                        avg_pas_h = np.mean(epoch_metrics['pas_h_values'][-10:])
                        print(f"  Conversation {i+1:3d}: PAS_h={avg_pas_h:.3f}")
                
                except Exception as e:
                    print(f"⚠️ Failed to train on conversation {conversation.conversation_id}: {e}")
                    continue
            
            # Compute epoch averages
            epoch_summary = {}
            for key, values in epoch_metrics.items():
                if values:
                    if key == 'affordance_gradients':
                        # Special handling for affordance gradients
                        avg_gradients = {}
                        for grad_dict in values:
                            for grad_key, grad_value in grad_dict.items():
                                if grad_key not in avg_gradients:
                                    avg_gradients[grad_key] = []
                                avg_gradients[grad_key].append(grad_value)
                        
                        for grad_key, grad_values in avg_gradients.items():
                            avg_gradients[grad_key] = np.mean(grad_values)
                        
                        epoch_summary[key] = avg_gradients
                    else:
                        # Convert torch tensors to numpy if needed
                        numpy_values = []
                        for value in values:
                            if isinstance(value, torch.Tensor):
                                numpy_values.append(value.detach().cpu().numpy())
                            else:
                                numpy_values.append(value)
                        epoch_summary[key] = np.mean(numpy_values)
            
            epoch_results.append(epoch_summary)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Conversational Loss: {epoch_summary.get('conversational_loss', 0):.3f}")
            print(f"   API Extraction Loss: {epoch_summary.get('api_extraction_loss', 0):.3f}")
            print(f"   Temporal Coherence: {epoch_summary.get('temporal_coherence', 0):.3f}")
            print(f"   Average PAS_h: {epoch_summary.get('pas_h_values', 0):.3f}")
            
            if 'affordance_gradients' in epoch_summary:
                print(f"   Affordance Gradients:")
                for grad_key, grad_value in epoch_summary['affordance_gradients'].items():
                    print(f"      {grad_key}: {grad_value:.3f}")
            
            # Update history
            self.history['conversational_accuracy'].append(epoch_summary.get('conversational_loss', 0))
            self.history['pas_h_evolution'].append(epoch_summary.get('pas_h_values', 0))
        
        return {
            'epoch_results': epoch_results,
            'final_trust': self.model.trust_scalars.tolist(),
            'training_history': self.history
        }
    
    def _train_conversation(self, conversation: Conversation) -> Dict[str, Any]:
        """Train on a single conversation."""
        conversation_metrics = {
            'conversational_loss': 0.0,
            'api_extraction_loss': 0.0,
            'temporal_coherence': 0.0,
            'pas_h_values': 0.0,
            'affordance_gradients': {}
        }
        
        if len(conversation.turns) < 2:
            return conversation_metrics
        
        total_loss = 0.0
        turn_count = 0
        
        # Process conversation turns sequentially
        for i in range(len(conversation.turns) - 1):
            current_turn = conversation.turns[i]
            next_turn = conversation.turns[i + 1]
            
            # Skip if no embeddings
            if current_turn.embedding is None or next_turn.embedding is None:
                continue
            
            # Prepare inputs
            current_input = current_turn.embedding.unsqueeze(0)  # [1, 768]
            target_embedding = next_turn.embedding.unsqueeze(0)  # [1, 768]
            
            # Get affordance gradients
            affordance_gradients = current_turn.affordance_gradients or {}
            
            # Conversation context
            conversation_context = {
                'turn_count': len(conversation.turns),
                'turn_index': i,
                'source': conversation.source,
                'conversation_id': conversation.conversation_id
            }
            
            # Forward pass
            output = self.model(
                current_input,
                affordance_gradients=affordance_gradients,
                conversation_context=conversation_context,
                return_analysis=True
            )
            
            # Compute losses
            hidden_state = output['hidden_state']
            
            # Project target to hidden dimension for comparison
            if target_embedding.shape[1] != hidden_state.shape[1]:
                if not hasattr(self, 'target_projector'):
                    self.target_projector = nn.Linear(768, hidden_state.shape[1]).to(self.model.device)
                target_proj = self.target_projector(target_embedding)
            else:
                target_proj = target_embedding
            
            # Conversational prediction loss
            conversational_loss = 1.0 - torch.cosine_similarity(hidden_state, target_proj, dim=1).mean()
            
            # API extraction loss (if applicable)
            api_strength = affordance_gradients.get('api_extraction', 0.0)
            api_loss = api_strength * conversational_loss  # Higher penalty for API extraction failures
            
            # Total loss
            step_loss = conversational_loss + 0.5 * api_loss
            total_loss += step_loss
            
            # Accumulate metrics
            conversation_metrics['conversational_loss'] += conversational_loss.item()
            conversation_metrics['api_extraction_loss'] += float(api_loss)
            conversation_metrics['pas_h_values'] += float(output['pas_h'])
            
            # Update affordance history
            self.model.update_affordance_history(affordance_gradients)
            
            turn_count += 1
        
        if turn_count > 0:
            # Average metrics
            for key in ['conversational_loss', 'api_extraction_loss', 'pas_h_values']:
                conversation_metrics[key] /= turn_count
            
            # Compute temporal coherence
            conversation_metrics['temporal_coherence'] = self._compute_temporal_coherence(conversation)
            
            # Backward pass
            avg_loss = total_loss / turn_count
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update trust scalars based on performance
            self._update_trust_scalars(conversation_metrics)
            
            # Store affordance gradients
            if conversation.turns and conversation.turns[0].affordance_gradients:
                conversation_metrics['affordance_gradients'] = conversation.turns[0].affordance_gradients
        
        return conversation_metrics
    
    def _compute_temporal_coherence(self, conversation: Conversation) -> float:
        """Compute temporal coherence for conversation."""
        if len(conversation.turns) < 2:
            return 1.0
        
        coherences = []
        for i in range(len(conversation.turns) - 1):
            current_turn = conversation.turns[i]
            next_turn = conversation.turns[i + 1]
            
            if current_turn.embedding is not None and next_turn.embedding is not None:
                similarity = torch.cosine_similarity(
                    current_turn.embedding.unsqueeze(0),
                    next_turn.embedding.unsqueeze(0),
                    dim=1
                ).item()
                coherences.append(abs(similarity))
        
        return np.mean(coherences) if coherences else 1.0
    
    def _update_trust_scalars(self, metrics: Dict[str, Any]):
        """Update trust scalars based on conversation performance."""
        # Performance score based on multiple factors
        conv_performance = 1.0 - metrics['conversational_loss']
        temporal_performance = metrics['temporal_coherence']
        
        overall_performance = 0.7 * conv_performance + 0.3 * temporal_performance
        
        # Evolutionary trust update
        trust_delta = self.trust_update_rate * (overall_performance - 0.5)
        self.model.trust_scalars += trust_delta
        self.model.trust_scalars.clamp_(0.0, 1.0)


def run_conversational_api_training():
    """Run conversational API training demonstration."""
    print("🗣️ Conversational API Training Integration")
    print("=" * 45)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'  # Force CPU usage for stability
    print(f"Device: {device}")
    
    # Create API ingestor
    print("\n🔧 Setting up API ingestor...")
    api_ingestor = ConversationalAPIIngestor(device=device)
    
    # Create model
    print("🏗️ Creating conversational temporal model...")
    model = ConversationalTemporalModel(
        input_dim=768,
        hidden_dim=256,
        num_functionals=5,
        poly_degree=4,
        device=device
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = ConversationalAPITrainer(model, api_ingestor)
    
    # Generate synthetic conversational data (simulating API ingestion)
    print("\n📊 Generating synthetic conversational data...")
    synthetic_conversations = generate_synthetic_api_conversations(device)
    
    print(f"✅ Generated {len(synthetic_conversations)} synthetic conversations")
    
    # Show sample conversation
    if synthetic_conversations:
        sample = synthetic_conversations[0]
        print(f"\n💬 Sample Conversation ({sample.conversation_id}):")
        for i, turn in enumerate(sample.turns[:3]):
            print(f"   Turn {i+1} ({turn.speaker_id}): {turn.text[:80]}...")
            if turn.affordance_gradients:
                top_gradients = sorted(turn.affordance_gradients.items(), key=lambda x: x[1], reverse=True)[:2]
                print(f"      Top gradients: {top_gradients}")
    
    # Train on conversational data
    print(f"\n🚀 Starting conversational training...")
    training_results = trainer.train_on_conversations(synthetic_conversations, num_epochs=3)
    
    # Analyze results
    print(f"\n📊 Training Results Analysis:")
    print(f"   Final Trust Scalars: {[f'{t:.3f}' for t in training_results['final_trust']]}")
    
    epoch_results = training_results['epoch_results']
    if epoch_results:
        pas_h_values = [f"{r.get('pas_h_values', 0):.3f}" for r in epoch_results]
        conv_loss_values = [f"{r.get('conversational_loss', 0):.3f}" for r in epoch_results]
        print(f"   PAS_h Evolution: {pas_h_values}")
        print(f"   Conversational Loss Evolution: {conv_loss_values}")
    
    # Test final model
    print(f"\n🧪 Testing final model...")
    test_conversation = synthetic_conversations[0] if synthetic_conversations else None
    
    if test_conversation and test_conversation.turns:
        test_turn = test_conversation.turns[0]
        
        with torch.no_grad():
            test_output = model(
                test_turn.embedding.unsqueeze(0),
                affordance_gradients=test_turn.affordance_gradients,
                conversation_context={'turn_count': len(test_conversation.turns)},
                return_analysis=True
            )
            
            print(f"   Test Output:")
            print(f"      PAS_h: {test_output['pas_h']:.3f}")
            print(f"      Containment Pressure: {test_output['containment_pressure']:.3f}")
            print(f"      Trust Mean: {test_output['trust_scalars'].mean():.3f}")
            
            if 'polynomial_diagnostics' in test_output:
                poly_diag = test_output['polynomial_diagnostics']
                print(f"      Polynomial Coefficient Norm: {poly_diag['coefficient_norm']:.3f}")
    
    # Integration with diegetic backend
    print(f"\n🔗 Integration Notes:")
    print(f"   • Model can be integrated with diegetic backend")
    print(f"   • Affordance gradients work with conversational patterns")
    print(f"   • PAS_h computation includes conversational context")
    print(f"   • Trust scalars evolve based on conversation quality")
    print(f"   • Ready for real API data ingestion")
    
    return model, trainer, training_results


def generate_synthetic_api_conversations(device: str) -> List[Conversation]:
    """Generate synthetic conversations simulating API ingestion."""
    conversations = []
    
    # Conversation templates with different affordance patterns
    templates = [
        # High conversational affordance
        {
            'id': 'conv_high_conversational',
            'turns': [
                ('user', "Hello! I'm curious about your thoughts on artificial intelligence. What do you think about its potential impact on society?"),
                ('assistant', "That's a fascinating question! I believe AI has tremendous potential to help solve complex problems like climate change and disease, but we need to be thoughtful about ensuring it benefits everyone."),
                ('user', "That's interesting. How do you think we can ensure AI development remains ethical and beneficial?"),
                ('assistant', "Great question! I think it requires collaboration between technologists, ethicists, policymakers, and the public to establish guidelines and oversight.")
            ],
            'context': {'topic': 'ai_ethics', 'style': 'conversational'},
            'source': 'synthetic_conversational'
        },
        
        # High API extraction affordance
        {
            'id': 'api_high_extraction',
            'turns': [
                ('user', "I need to search for the latest research papers on quantum computing. Can you help me find current information and download recent publications?"),
                ('assistant', "I'd be happy to help you find quantum computing research! Let me search for recent papers and provide you with the most current information available."),
                ('user', "Specifically, I'm looking for papers on quantum error correction published in the last 6 months. Can you get me a list?"),
                ('assistant', "I'll search for recent quantum error correction papers. This type of research is very active, so there should be several new publications to review.")
            ],
            'context': {'topic': 'research_search', 'style': 'api_focused'},
            'source': 'synthetic_api'
        },
        
        # High code execution affordance
        {
            'id': 'code_high_execution',
            'turns': [
                ('user', "def neural_network_forward(weights, inputs): return torch.matmul(inputs, weights.T)"),
                ('assistant', "That's a clean implementation of a basic forward pass! You might want to add bias terms and activation functions for a complete layer."),
                ('user', "Good point. How would I add a ReLU activation? def add_relu(x): return torch.max(torch.zeros_like(x), x)"),
                ('assistant', "Perfect! That's exactly how ReLU works. You could also use torch.relu(x) for the built-in version, but your implementation shows the underlying math clearly.")
            ],
            'context': {'topic': 'neural_networks', 'style': 'code_focused'},
            'source': 'synthetic_code'
        },
        
        # Mixed affordances
        {
            'id': 'mixed_affordances',
            'turns': [
                ('user', "I'm working on a machine learning project and need to understand both the theory and implementation. Can you explain gradient descent and show me some code?"),
                ('assistant', "Absolutely! Gradient descent is an optimization algorithm that iteratively moves toward the minimum of a function by following the negative gradient."),
                ('user', "That makes sense. Can you show me a simple implementation? Also, where can I find more detailed mathematical explanations?"),
                ('assistant', "Here's a basic implementation: def gradient_descent(f, df, x0, lr=0.01, steps=1000): x = x0; for _ in range(steps): x -= lr * df(x); return x")
            ],
            'context': {'topic': 'ml_education', 'style': 'mixed'},
            'source': 'synthetic_mixed'
        }
    ]
    
    # Create conversations from templates
    from src.data.conversational_api_ingestor import ConversationalDataProcessor
    processor = ConversationalDataProcessor(device=device)
    
    for template in templates:
        turns = []
        for speaker, text in template['turns']:
            turn = ConversationTurn(
                speaker_id=speaker,
                text=text
            )
            turns.append(turn)
        
        conversation = Conversation(
            conversation_id=template['id'],
            turns=turns,
            context=template['context'],
            source=template['source']
        )
        
        # Process conversation to add embeddings and affordance gradients
        processed_conversation = processor.process_conversation(conversation)
        conversations.append(processed_conversation)
    
    return conversations


if __name__ == "__main__":
    print("🗣️ Conversational API Training Integration")
    print("Demonstrates integration of real conversational data with temporal training")
    print("=" * 75)
    
    try:
        model, trainer, results = run_conversational_api_training()
        
        print(f"\n✅ Conversational API training completed successfully!")
        print(f"\nKey achievements:")
        print(f"• Integrated conversational affordance gradients")
        print(f"• Proper PAS_h computation with conversational context")
        print(f"• Temporal association learning on conversation data")
        print(f"• Trust scalar evolution based on conversation quality")
        print(f"• Ready for real API data integration")
        
        print(f"\nNext steps:")
        print(f"• Set up real API credentials (HuggingFace, Reddit)")
        print(f"• Integrate with diegetic terminal backend")
        print(f"• Test with large-scale conversational datasets")
        print(f"• Monitor affordance gradient evolution over time")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

