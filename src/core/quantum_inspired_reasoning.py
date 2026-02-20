import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class QuantumInspiredReasoningState(nn.Module):
    """
    Quantum-inspired reasoning mechanism using complex-valued tensors (System 2 Extension).
    Modeling superposition of hypotheses and concept entanglement.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        # Initialize complex amplitude state |ψ⟩
        real_part = torch.randn(dim)
        imag_part = torch.randn(dim)
        self.amplitude = torch.complex(real_part, imag_part)
        self.amplitude = self.amplitude / (torch.norm(self.amplitude) + 1e-8)
        
        # Hamiltonian for evolution (Hermitian)
        H_real = torch.randn(dim, dim)
        self.reasoning_hamiltonian = torch.complex(H_real, torch.zeros_like(H_real))
        # Make Hermitian: H = (A + A^H) / 2
        self.reasoning_hamiltonian = (self.reasoning_hamiltonian + self.reasoning_hamiltonian.conj().T) * 0.5
        
    def superposition_reasoning(self, hypotheses: List[torch.Tensor]) -> torch.Tensor:
        """
        Reason over a list of hypothesis vectors by superposing them.
        Returns: Probability distribution over dimensions (Interpretation).
        Supports batched inputs [Batch, Dim] or single inputs [Dim].
        """
        if not hypotheses:
            return torch.zeros(self.dim)
            
        # 1. Determine Batch Shape
        # Assume all hypotheses have same batch shape
        ref_shape = hypotheses[0].shape
        is_batched = len(ref_shape) > 1
        batch_dim = ref_shape[0] if is_batched else 1
        
        # 2. Create Superposition State |S⟩ = Σ c_i |h_i⟩
        if is_batched:
             super_state = torch.zeros(ref_shape, dtype=torch.complex64, device=self.amplitude.device)
        else:
             super_state = torch.zeros(self.dim, dtype=torch.complex64, device=self.amplitude.device)
             
        n = len(hypotheses)
        coeff = 1.0 / np.sqrt(n)
        
        for h in hypotheses:
            # Ensure shape match (pad or slice last dim)
            if h.shape[-1] != self.dim:
                if h.shape[-1] > self.dim: 
                    h = h[..., :self.dim]
                else: 
                    padding = (0, self.dim - h.shape[-1])
                    h = torch.nn.functional.pad(h, padding)
                
            # Map real hypothesis to complex state
            complex_h = torch.complex(h, torch.zeros_like(h))
            super_state += coeff * complex_h
            
        # 2. Evolve state: |S(t)⟩ = e^{-iHt} |S(0)⟩ (1 step)
        dt = 0.1
        # Evolution operator U = exp(-iHt) ~ (I - iH*dt)
        evolution = torch.eye(self.dim, dtype=torch.complex64, device=self.amplitude.device) - \
                   1j * self.reasoning_hamiltonian * dt
                   
        # Evolution Logic:
        # If State is [dim] -> U @ S
        # If State is [B, dim] -> S @ U.T
        if is_batched:
            evolved_state = torch.matmul(super_state, evolution.T)
        else:
            evolved_state = torch.matmul(evolution, super_state)
        
        # Re-normalize
        norm = torch.norm(evolved_state, dim=-1, keepdim=True)
        evolved_state = evolved_state / (norm + 1e-8)
        
        # 3. Born Rule: P(x) = |ψ(x)|²
        probabilities = torch.abs(evolved_state) ** 2
        
        return probabilities

    def entangle_concepts(self, concept_a: torch.Tensor, concept_b: torch.Tensor) -> torch.Tensor:
        """
        Create an entangled state between two concepts (Tensor Product).
        Returns entangled tensor (flattened or shaped).
        """
        # Tensor product: A ⊗ B
        # For computability, if dims are large, we use outer product simulation
        flat_a = concept_a.flatten()
        flat_b = concept_b.flatten()
        
        # Limit dimension for demo
        max_ent_dim = 256
        if flat_a.numel() * flat_b.numel() > max_ent_dim:
             # Compressed entanglement (element-wise + cross)
             len_min = min(flat_a.numel(), flat_b.numel())
             entangled = flat_a[:len_min] * flat_b[:len_min] # Very simplified trace
             return entangled
             
        entangled_matrix = torch.outer(flat_a, flat_b)
        return entangled_matrix
        
    def quantum_measurement(self, state: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Simulate measurement.
        Returns (Expectation Value <O>, Collapsed State).
        """
        # Observable O is assumed to be Position (Identity in this basis)
        prob = torch.abs(state) ** 2
        prob = prob / prob.sum()
        
        # Collapse to a specific eigenstate based on probability
        indices = torch.arange(len(prob))
        collapsed_idx = indices[torch.multinomial(prob, 1)].item()
        
        collapsed_state = torch.zeros_like(state)
        collapsed_state[collapsed_idx] = 1.0 + 0j
        
        expectation = torch.sum(indices * prob).item()
        
        return expectation, collapsed_state
        
    def decoherence_model(self, state: torch.Tensor, noise_strength: float = 0.1) -> torch.Tensor:
        """Mix state with max-entropy noise."""
        noise = torch.complex(torch.randn_like(state.real), torch.randn_like(state.real))
        noise = noise / torch.norm(noise)
        
        # ρ' = (1-p)ρ + p(I/d)
        # Vector approximation
        decoherent = (1 - noise_strength) * state + noise_strength * noise
        return decoherent / torch.norm(decoherent)

    def quantum_interference(self, state_a: torch.Tensor, state_b: torch.Tensor, phase_shift: float) -> torch.Tensor:
        """
        Interference: |ψ⟩ = |a⟩ + e^{iφ}|b⟩
        """
        return state_a + state_b * np.exp(1j * phase_shift)

    def update_hamiltonian(self, gradient: torch.Tensor, learning_rate: float = 0.001):
        """Update Hamiltonian logic based on reasoning feedback."""
        # Must ensure H stays Hermitian
        grad_complex = torch.complex(gradient, torch.zeros_like(gradient))
        update = (grad_complex + grad_complex.conj().T) * 0.5
        
        with torch.no_grad():
            self.reasoning_hamiltonian -= learning_rate * update