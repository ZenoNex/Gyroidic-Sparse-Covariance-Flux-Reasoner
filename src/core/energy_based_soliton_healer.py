"""
Energy-Based Soliton Stability Healer
Based on "A Tutorial on Energy-Based Learning" principles.

Implements soliton preservation using:
1. Energy minimization for stable configurations
2. Margin-based loss functions for robustness
3. Constraint satisfaction through energy shaping
4. Number-theoretic stability via modular arithmetic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class EnergyBasedSolitonHealer(nn.Module):
    """
    Energy-based soliton healer following EBM principles.
    
    Key concepts from energy-based learning:
    - Energy E(W, Y, X) measures compatibility
    - Lower energy = more compatible/stable configuration
    - Learning shapes energy surface for desired outcomes
    """
    
    def __init__(self, 
                 state_dim: int = 64,
                 energy_margin: float = 1.0,
                 healing_rate: float = 0.1,
                 stability_threshold: float = 1e-3):
        super().__init__()
        
        self.state_dim = state_dim
        self.energy_margin = energy_margin
        self.healing_rate = healing_rate
        self.stability_threshold = stability_threshold
        
        # Energy function parameters (learnable)
        self.energy_weights = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.energy_bias = nn.Parameter(torch.zeros(state_dim))
        
        # Soliton template (stable configuration)
        self.register_buffer('soliton_template', self._create_soliton_template())
        
        # Healing history for adaptive learning
        self.healing_history = []
        self.max_history = 100
    
    @staticmethod
    def _generate_primes(n: int) -> list:
        """Generate the first n primes dynamically (no hardcoded lists)."""
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
                if p * p > candidate:
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    def _create_soliton_template(self) -> torch.Tensor:
        """
        Create stable soliton template based on number theory.
        Uses golden ratio and dynamically generated prime-based spacing for stability.
        """
        # Golden ratio for natural stability
        phi = (1 + np.sqrt(5)) / 2
        
        # Create template with golden ratio spacing
        indices = torch.arange(self.state_dim, dtype=torch.float32)
        template = torch.cos(2 * np.pi * indices / phi)
        
        # Add prime-based modulation for number-theoretic stability
        # Dynamically generated — no hardcoded prime lists (anti-lobotomy compliance)
        num_primes = min(10, self.state_dim // 8)
        primes = torch.tensor(self._generate_primes(max(num_primes, 1)), dtype=torch.float32)
        for i, p in enumerate(primes[:num_primes]):
            modulation = torch.sin(2 * np.pi * indices / p) * 0.1
            template += modulation
        
        # Normalize to unit energy
        template = template / torch.norm(template)
        
        return template
    
    def compute_energy(self, state: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy E(W, Y, X) for given state.
        Lower energy indicates more stable/compatible configuration.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        
        if target is None:
            # Use soliton template as target
            target = self.soliton_template.unsqueeze(0).expand(batch_size, -1)
        
        # Energy based on distance from stable configuration
        # E(state, target) = ||A(state - target)||² + bias·state
        diff = state - target
        quadratic_energy = torch.sum(diff * torch.mm(diff, self.energy_weights), dim=1)
        linear_energy = torch.sum(state * self.energy_bias.unsqueeze(0), dim=1)
        
        total_energy = quadratic_energy + linear_energy
        
        return total_energy
    
    def compute_energy_gradient(self, state: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient of energy function for healing direction.
        """
        state.requires_grad_(True)
        energy = self.compute_energy(state, target)
        
        # Compute gradient
        grad = torch.autograd.grad(energy.sum(), state, create_graph=True)[0]
        
        return grad
    
    def heal_soliton(self, state: torch.Tensor, iteration_count: int = 1) -> Tuple[torch.Tensor, Dict]:
        """
        Heal soliton using energy-based gradient descent.
        
        Implements margin-based healing:
        - If energy > margin, apply strong healing
        - If energy < margin, apply gentle stabilization
        """
        original_shape = state.shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        healed_state = state.clone()
        
        diagnostics = {
            'initial_energy': [],
            'final_energy': [],
            'healing_steps': [],
            'stability_achieved': []
        }
        
        for batch_idx in range(batch_size):
            current_state = healed_state[batch_idx:batch_idx+1]
            
            # Compute initial energy
            initial_energy = self.compute_energy(current_state).item()
            diagnostics['initial_energy'].append(initial_energy)
            
            healing_steps = 0
            for step in range(iteration_count * 10):  # Max healing steps
                # Compute energy and gradient
                energy = self.compute_energy(current_state)
                
                # Check if healing is needed
                if energy.item() < self.energy_margin:
                    # Gentle stabilization
                    healing_rate = self.healing_rate * 0.1
                else:
                    # Strong healing
                    healing_rate = self.healing_rate
                
                # Compute healing direction (negative gradient)
                grad = self.compute_energy_gradient(current_state)
                healing_direction = -grad
                
                # Apply healing with adaptive rate
                current_state = current_state + healing_rate * healing_direction
                
                # Numerical stabilization
                current_state = torch.clamp(current_state, -10.0, 10.0)
                
                healing_steps += 1
                
                # Check convergence
                if torch.norm(healing_direction).item() < self.stability_threshold:
                    break
            
            # Update healed state
            healed_state[batch_idx:batch_idx+1] = current_state
            
            # Final diagnostics
            final_energy = self.compute_energy(current_state).item()
            diagnostics['final_energy'].append(final_energy)
            diagnostics['healing_steps'].append(healing_steps)
            diagnostics['stability_achieved'].append(final_energy < self.energy_margin)
        
        # Restore original shape
        if len(original_shape) == 1:
            healed_state = healed_state.squeeze(0)
        
        # Update healing history
        self.healing_history.append({
            'avg_initial_energy': np.mean(diagnostics['initial_energy']),
            'avg_final_energy': np.mean(diagnostics['final_energy']),
            'avg_healing_steps': np.mean(diagnostics['healing_steps']),
            'stability_rate': np.mean(diagnostics['stability_achieved'])
        })
        
        # Trim history
        if len(self.healing_history) > self.max_history:
            self.healing_history = self.healing_history[-self.max_history:]
        
        # Aggregate diagnostics
        aggregate_diagnostics = {
            'alpha': 1.0 + iteration_count * 0.001,  # Adaptive healing strength
            'healing_progress': np.mean(diagnostics['stability_achieved']) if diagnostics['stability_achieved'] else 0.0,
            'iteration_count': iteration_count,
            'avg_energy_reduction': (np.mean(diagnostics['initial_energy']) - np.mean(diagnostics['final_energy'])) if diagnostics['initial_energy'] and diagnostics['final_energy'] else 0.0,
            'convergence_rate': 1.0 - (np.mean(diagnostics['healing_steps']) / (iteration_count * 10)) if diagnostics['healing_steps'] else 0.0,
            'initial_energy': diagnostics['initial_energy'],
            'final_energy': diagnostics['final_energy'],
            'healing_steps': diagnostics['healing_steps'],
            'stability_achieved': diagnostics['stability_achieved']
        }
        
        return healed_state, aggregate_diagnostics
    
    def update_energy_function(self, positive_states: torch.Tensor, negative_states: torch.Tensor):
        """
        Update energy function using contrastive learning.
        
        Implements hinge loss from energy-based learning:
        L = max(0, margin + E(positive) - E(negative))
        """
        if len(positive_states) == 0 or len(negative_states) == 0:
            return
        
        # Compute energies
        pos_energy = self.compute_energy(positive_states)
        neg_energy = self.compute_energy(negative_states)
        
        # Hinge loss with margin
        loss = torch.clamp(self.energy_margin + pos_energy.mean() - neg_energy.mean(), min=0)
        
        if loss.item() > 0:
            # Update parameters to reduce loss
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Simple SGD update
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param -= 0.01 * param.grad
                        param.grad.zero_()
    
    def get_stability_metrics(self) -> Dict:
        """Get current stability metrics."""
        if not self.healing_history:
            return {'stability_score': 0.0, 'healing_efficiency': 0.0}
        
        recent_history = self.healing_history[-10:]  # Last 10 healings
        
        stability_score = np.mean([h['stability_rate'] for h in recent_history])
        healing_efficiency = np.mean([h['convergence_rate'] for h in recent_history])
        
        return {
            'stability_score': stability_score,
            'healing_efficiency': healing_efficiency,
            'avg_energy_level': np.mean([h['avg_final_energy'] for h in recent_history]),
            'healing_consistency': 1.0 - np.std([h['avg_final_energy'] for h in recent_history])
        }

def create_energy_based_healer(state_dim: int = 64) -> EnergyBasedSolitonHealer:
    """Factory function to create energy-based soliton healer."""
    return EnergyBasedSolitonHealer(
        state_dim=state_dim,
        energy_margin=1.0,
        healing_rate=0.1,
        stability_threshold=1e-3
    )
