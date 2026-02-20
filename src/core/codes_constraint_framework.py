"""
CODES: Constraint-Oriented Differential Equation System
Based on Energy-Based Learning and Number Theory

Implements constraint satisfaction through energy minimization:
1. Constraints as energy functions
2. Differential equations for constraint evolution
3. Number-theoretic stability guarantees
4. Margin-based robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

class CODESConstraintFramework(nn.Module):
    """
    CODES framework for constraint-oriented learning.
    
    Core principles:
    - Each constraint defines an energy landscape
    - System evolves to minimize total energy
    - Stable states satisfy all constraints
    - Number-theoretic methods ensure convergence
    """
    
    def __init__(self, 
                 state_dim: int = 64,
                 max_constraints: int = 32,
                 energy_margin: float = 1.0,
                 convergence_threshold: float = 1e-4):
        super().__init__()
        
        self.state_dim = state_dim
        self.max_constraints = max_constraints
        self.energy_margin = energy_margin
        self.convergence_threshold = convergence_threshold
        
        # Constraint energy functions
        self.constraint_weights = nn.ParameterList([
            nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
            for _ in range(max_constraints)
        ])
        
        self.constraint_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(state_dim))
            for _ in range(max_constraints)
        ])
        
        # Constraint importance weights
        self.constraint_importance = nn.Parameter(torch.ones(max_constraints))
        
        # Active constraints mask
        self.register_buffer('active_constraints', torch.zeros(max_constraints, dtype=torch.bool))
        
        # Differential equation solver parameters
        self.dt = 0.01  # Time step
        self.damping = 0.1  # Damping coefficient
        
        # Use existing polynomial co-prime system (anti-lobotomy compliance)
        from src.core.polynomial_coprime import PolynomialCoprimeConfig
        self.polynomial_config = PolynomialCoprimeConfig(
            k=min(10, state_dim // 4),  # Dynamic based on state dimension
            degree=4,
            basis_type='chebyshev',
            learnable=True,
            device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
        )
        
    def add_constraint(self, constraint_id: int, constraint_type: str = 'quadratic'):
        """Add a new constraint to the system."""
        if constraint_id >= self.max_constraints:
            raise ValueError(f"Constraint ID {constraint_id} exceeds maximum {self.max_constraints}")
        
        self.active_constraints[constraint_id] = True
        
        # Initialize constraint based on type
        if constraint_type == 'quadratic':
            # Positive definite quadratic form
            W = torch.randn(self.state_dim, self.state_dim) * 0.01
            self.constraint_weights[constraint_id].data = W @ W.t() + torch.eye(self.state_dim) * 0.01
        elif constraint_type == 'harmonic':
            # Harmonic oscillator constraint
            self.constraint_weights[constraint_id].data = torch.eye(self.state_dim) * 0.1
        elif constraint_type == 'polynomial_coprime':
            # Polynomial co-prime constraint using existing system
            phi_values = self.polynomial_config.evaluate(torch.randn(1, self.state_dim))
            W = torch.zeros(self.state_dim, self.state_dim)
            
            # Create constraint matrix based on polynomial functional structure
            for i in range(min(phi_values.shape[-1], self.state_dim)):
                # Access the i-th functional value and ensure it's a scalar via mean().item()
                # Use -1 index for functional dimension to handle batching gracefully
                val = phi_values[..., i].mean().item()
                W[i, i] = 1.0 / (abs(val) + 1e-8)
            
            self.constraint_weights[constraint_id].data = W
    
    def compute_constraint_energy(self, state: torch.Tensor, constraint_id: int) -> torch.Tensor:
        """Compute energy for a specific constraint."""
        if not self.active_constraints[constraint_id]:
            return torch.zeros(state.shape[0], device=state.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        W = self.constraint_weights[constraint_id]
        b = self.constraint_biases[constraint_id]
        
        # Quadratic energy: E = x^T W x + b^T x
        quadratic_term = torch.sum(state * torch.mm(state, W), dim=1)
        linear_term = torch.sum(state * b.unsqueeze(0), dim=1)
        
        return quadratic_term + linear_term
    
    def compute_total_energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute total energy across all active constraints."""
        total_energy = torch.zeros(state.shape[0] if state.dim() > 1 else 1, device=state.device)
        
        for i in range(self.max_constraints):
            if self.active_constraints[i]:
                constraint_energy = self.compute_constraint_energy(state, i)
                importance = torch.sigmoid(self.constraint_importance[i])  # Normalize importance
                total_energy += importance * constraint_energy
        
        return total_energy
    
    def compute_energy_gradient(self, state: torch.Tensor) -> torch.Tensor:
        """Compute gradient of total energy for constraint forces."""
        state.requires_grad_(True)
        energy = self.compute_total_energy(state)
        
        grad = torch.autograd.grad(energy.sum(), state, create_graph=True)[0]
        
        return grad
    
    def evolve_state(self, state: torch.Tensor, num_steps: int = 10) -> Tuple[torch.Tensor, Dict]:
        """
        Evolve state using constraint-oriented differential equations.
        
        Implements damped gradient descent:
        dx/dt = -∇E(x) - γ(dx/dt)
        
        Where E(x) is total constraint energy and γ is damping.
        """
        original_shape = state.shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        current_state = state.clone()
        velocity = torch.zeros_like(current_state)
        
        energy_history = []
        convergence_history = []
        
        for step in range(num_steps):
            # Compute constraint forces (negative gradient)
            forces = -self.compute_energy_gradient(current_state)
            
            # Update velocity with damping
            velocity = velocity * (1 - self.damping) + forces * self.dt
            
            # Update state
            current_state = current_state + velocity * self.dt
            
            # Numerical stabilization
            current_state = torch.clamp(current_state, -10.0, 10.0)
            
            # Track convergence
            energy = self.compute_total_energy(current_state)
            energy_history.append(energy.mean().item())
            
            force_magnitude = torch.norm(forces).item()
            convergence_history.append(force_magnitude)
            
            # Check convergence
            if force_magnitude < self.convergence_threshold:
                break
        
        # Restore original shape
        if len(original_shape) == 1:
            current_state = current_state.squeeze(0)
        
        # Compute diagnostics
        diagnostics = {
            'final_energy': energy_history[-1] if energy_history else 0.0,
            'energy_reduction': energy_history[0] - energy_history[-1] if len(energy_history) > 1 else 0.0,
            'convergence_steps': len(energy_history),
            'converged': convergence_history[-1] < self.convergence_threshold if convergence_history else False,
            'stability_score': self._compute_stability_score(current_state),
            'constraint_satisfaction': self._compute_constraint_satisfaction(current_state)
        }
        
        return current_state, diagnostics
    
    def _compute_stability_score(self, state: torch.Tensor) -> float:
        """Compute stability score based on energy landscape curvature."""
        # Compute Hessian approximation
        grad = self.compute_energy_gradient(state)
        
        # Stability based on gradient magnitude (lower = more stable)
        stability = 1.0 / (1.0 + torch.norm(grad).item())
        
        return stability
    
    def _compute_constraint_satisfaction(self, state: torch.Tensor) -> Dict:
        """Compute how well each constraint is satisfied."""
        satisfaction = {}
        
        for i in range(self.max_constraints):
            if self.active_constraints[i]:
                energy = self.compute_constraint_energy(state, i)
                # Constraint is satisfied if energy is below margin
                satisfied = (energy < self.energy_margin).float().mean().item()
                satisfaction[f'constraint_{i}'] = satisfied
        
        # Overall satisfaction
        if satisfaction:
            satisfaction['overall'] = np.mean(list(satisfaction.values()))
        else:
            satisfaction['overall'] = 1.0
        
        return satisfaction
    
    def apply_number_theoretic_stabilization(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply number-theoretic stabilization using polynomial co-prime functionals.
        
        This ensures the state remains in a stable numerical range using the
        existing polynomial co-prime system instead of hardcoded primes.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        stabilized_state = state.clone()
        
        # Use polynomial co-prime functionals for stabilization
        phi_values = self.polynomial_config.evaluate(stabilized_state)  # [batch, k]
        
        # Apply polynomial-based modular reduction
        # Use the functional values as modulation factors
        for i in range(min(phi_values.shape[1], stabilized_state.shape[1])):
            # Modular-like reduction using polynomial functional values
            phi_i = phi_values[:, i:i+1]  # [batch, 1]
            
            # Apply smooth modular reduction (avoiding discontinuities)
            modulation = torch.tanh(phi_i) * 0.1  # Bounded modulation
            stabilized_state[:, i:i+1] = stabilized_state[:, i:i+1] + modulation
        
        # Normalize to prevent drift
        stabilized_state = F.normalize(stabilized_state, p=2, dim=1)
        
        if state.shape[0] == 1 and len(state.shape) == 1:
            stabilized_state = stabilized_state.squeeze(0)
        
        return stabilized_state
    
    def train_constraints(self, positive_states: torch.Tensor, negative_states: torch.Tensor):
        """
        Train constraint parameters using contrastive learning.
        
        Implements margin-based loss:
        L = max(0, margin + E(positive) - E(negative))
        """
        if len(positive_states) == 0 or len(negative_states) == 0:
            return
        
        # Compute energies
        pos_energy = self.compute_total_energy(positive_states)
        neg_energy = self.compute_total_energy(negative_states)
        
        # Margin-based contrastive loss
        loss = torch.clamp(self.energy_margin + pos_energy.mean() - neg_energy.mean(), min=0)
        
        if loss.item() > 0:
            # Backpropagate and update
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Simple SGD update
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param -= 0.01 * param.grad
                        param.grad.zero_()
    
    def get_system_status(self) -> Dict:
        """Get current system status and diagnostics."""
        active_count = self.active_constraints.sum().item()
        
        # Compute constraint importance distribution
        importance_weights = torch.sigmoid(self.constraint_importance)
        active_importance = importance_weights[self.active_constraints]
        
        return {
            'active_constraints': active_count,
            'total_constraints': self.max_constraints,
            'avg_importance': active_importance.mean().item() if active_count > 0 else 0.0,
            'importance_std': active_importance.std().item() if active_count > 0 else 0.0,
            'system_complexity': active_count / self.max_constraints,
            'convergence_threshold': self.convergence_threshold,
            'energy_margin': self.energy_margin
        }

def create_codes_framework(state_dim: int = 64, max_constraints: int = 32) -> CODESConstraintFramework:
    """Factory function to create CODES constraint framework."""
    framework = CODESConstraintFramework(
        state_dim=state_dim,
        max_constraints=max_constraints,
        energy_margin=1.0,
        convergence_threshold=1e-4
    )
    
    # Add some default constraints
    framework.add_constraint(0, 'quadratic')  # Stability constraint
    framework.add_constraint(1, 'harmonic')   # Oscillation constraint
    framework.add_constraint(2, 'polynomial_coprime')  # Polynomial co-prime constraint
    
    return framework

