"""
Quantum-inspired reasoning state representation.

This module models koncept superposition and entanglement using complex-valued
tensors and Hermitian Hamiltonian dynamics for System 2 reasoners.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from src.core.honest_jitter import harvest_honest_jitter

class QuantumInspiredReasoningState(nn.Module):
    """
    Quantum-inspired reasoning mechanism using complex-valued tensors (System 2 Extension).
    Modeling superposition of hypotheses and concept entanglement.
    """
    def __init__(self, dim: int = 256):
        """
        Initialize the QuantumInspiredReasoningState module.

        Args:
            dim: Dimension of the complex amplitude vector and square Hamiltonian.
        """
        super().__init__()
        self.dim = dim
        # Initialize complex amplitude state |
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        real_part = harvest_honest_jitter((dim,))
        imag_part = harvest_honest_jitter((dim,))
        self.amplitude = torch.complex(real_part, imag_part)
        self.amplitude = self.amplitude / (torch.norm(self.amplitude) + 1e-8)
        
        # Hamiltonian for evolution (Hermitian)
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        H_real = harvest_honest_jitter((dim, dim))
        self.reasoning_hamiltonian = torch.complex(H_real, torch.zeros_like(H_real))
        # Make Hermitian: H = (A + A^H) / 2
        self.reasoning_hamiltonian = (self.reasoning_hamiltonian + self.reasoning_hamiltonian.conj().T) * 0.5
        
    def superposition_reasoning(self, hypotheses: List[torch.Tensor], mode: str = 'PLAY') -> torch.Tensor:
        """
        Reason over a list of hypothesis vectors by superposing them.
        Returns: Probability distribution over dimensions (Interpretation).
        Supports batched inputs [Batch, Dim] or single inputs [Dim].
        """
        if not hypotheses:
            return torch.zeros(self.dim)
            
        is_play = mode.upper() in ('PLAY', 'GOO')
        is_serious = mode.upper() in ('SERIOUSNESS', 'PRICKLES')
        
        # 1. Determine Batch Shape
        # Assume all hypotheses have same batch shape
        ref_shape = hypotheses[0].shape
        is_batched = len(ref_shape) > 1
        batch_dim = ref_shape[0] if is_batched else 1
        
        # 2. Create Superposition State |S =  c_i |h_i
        if is_batched:
             super_state = torch.zeros((batch_dim, self.dim), dtype=torch.complex64, device=self.amplitude.device)
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
            
        # 2. Evolve state: |S(t) = e^{-iHt} |S(0) (1 step)
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
        
        if is_serious:
            # Collapse the superposition into a single logically consistent target residue via precise projective measurement
            probs_k = []
            for h in hypotheses:
                # Match shape of h with self.dim
                if h.shape[-1] != self.dim:
                    if h.shape[-1] > self.dim:
                        h = h[..., :self.dim]
                    else:
                        padding = (0, self.dim - h.shape[-1])
                        h = torch.nn.functional.pad(h, padding)
                
                # Normalize h
                h_norm = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-8)
                complex_h = torch.complex(h_norm, torch.zeros_like(h_norm))
                
                # Inner product with evolved_state
                if is_batched:
                    inner_prod = torch.sum(evolved_state * complex_h.conj(), dim=-1)
                else:
                    inner_prod = torch.sum(evolved_state * complex_h.conj())
                
                prob_h = torch.abs(inner_prod) ** 2
                probs_k.append(prob_h)
                
            probs_k_tensor = torch.stack(probs_k, dim=0) # [NumHypotheses, Batch] or [NumHypotheses]
            
            # Select the hypothesis with the maximum probability (collapse)
            if is_batched:
                best_indices = torch.argmax(probs_k_tensor, dim=0) # [Batch]
                collapsed_hypotheses = []
                for b in range(batch_dim):
                    best_idx = best_indices[b].item()
                    h_best = hypotheses[best_idx][b]
                    collapsed_hypotheses.append(h_best)
                collapsed_state = torch.stack(collapsed_hypotheses, dim=0) # [Batch, Dim]
            else:
                best_idx = torch.argmax(probs_k_tensor).item()
                collapsed_state = hypotheses[best_idx] # [Dim]
                
            # Ensure shape matches self.dim
            if collapsed_state.shape[-1] != self.dim:
                if collapsed_state.shape[-1] > self.dim:
                    collapsed_state = collapsed_state[..., :self.dim]
                else:
                    padding = (0, self.dim - collapsed_state.shape[-1])
                    collapsed_state = torch.nn.functional.pad(collapsed_state, padding)
            
            # Helper to project a vector to Birkhoff polytope
            def project_vector_to_birkhoff(v: torch.Tensor) -> torch.Tensor:
                orig_shape = v.shape
                dim_val = orig_shape[-1]
                n_val = int(np.sqrt(dim_val))
                if n_val * n_val != dim_val:
                    n_val = int(np.round(np.sqrt(dim_val)))
                    target_dim = n_val * n_val
                    if target_dim < dim_val:
                        v_p = v[..., :target_dim]
                    else:
                        padding = (0, target_dim - dim_val)
                        v_p = torch.nn.functional.pad(v, padding)
                else:
                    v_p = v
                    
                if v_p.dim() > 1:
                    batch_sz = v_p.shape[0]
                    v_mat = v_p.view(batch_sz, n_val, n_val)
                else:
                    v_mat = v_p.view(1, n_val, n_val)
                    
                from src.core.birkhoff_projection import project_to_birkhoff
                v_proj_mat = project_to_birkhoff(v_mat)
                
                if v_p.dim() > 1:
                    v_proj = v_proj_mat.view(batch_sz, -1)
                else:
                    v_proj = v_proj_mat.view(-1)
                    
                if v_proj.shape[-1] > dim_val:
                    v_proj = v_proj[..., :dim_val]
                elif v_proj.shape[-1] < dim_val:
                    padding = (0, dim_val - v_proj.shape[-1])
                    v_proj = torch.nn.functional.pad(v_proj, padding)
                    
                return v_proj
                
            # Project collapsed state onto Birkhoff Polytope
            probabilities = project_vector_to_birkhoff(collapsed_state)
            
        elif is_play:
            # Play mode decoherence model: noise >= 0.5
            # We must handle 1D vs 2D state
            if is_batched:
                decoherent_list = []
                for b in range(batch_dim):
                    dec = self.decoherence_model(evolved_state[b], noise_strength=0.5)
                    decoherent_list.append(dec)
                evolved_state = torch.stack(decoherent_list, dim=0)
            else:
                evolved_state = self.decoherence_model(evolved_state, noise_strength=0.5)
            probabilities = torch.abs(evolved_state) ** 2
        else:
            # 3. Born Rule: P(x) = |(x)|
            probabilities = torch.abs(evolved_state) ** 2
            
        return probabilities

    def entangle_concepts(self, concept_a: torch.Tensor, concept_b: torch.Tensor) -> torch.Tensor:
        """
        Create an entangled state between two concepts (Tensor Product).
        Returns entangled tensor (flattened or shaped).
        """
        # Tensor product: A  B
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
        # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
        noise_real = harvest_honest_jitter(state.shape)
        noise_imag = harvest_honest_jitter(state.shape)
        noise = torch.complex(noise_real, noise_imag)
        noise = noise / torch.norm(noise)
        
        # ' = (1-p) + p(I/d)
        # Vector approximation
        decoherent = (1 - noise_strength) * state + noise_strength * noise
        return decoherent / torch.norm(decoherent)

    def quantum_interference(self, state_a: torch.Tensor, state_b: torch.Tensor, phase_shift: float) -> torch.Tensor:
        """
        Interference: | = |a + e^{i}|b
        """
        return state_a + state_b * np.exp(1j * phase_shift)

    def update_hamiltonian(self, gradient: torch.Tensor, learning_rate: float = 0.001):
        """Update Hamiltonian logic based on reasoning feedback."""
        # Must ensure H stays Hermitian
        grad_complex = torch.complex(gradient, torch.zeros_like(gradient))
        update = (grad_complex + grad_complex.conj().T) * 0.5
        
        with torch.no_grad():
            self.reasoning_hamiltonian -= learning_rate * update

    def is_hermitian(self, tol: float = 1e-4):
        """
        Check whether reasoning Hamiltonian satisfies H = H^dagger.

        Returns:
            (is_herm, relative_hermitian_error)
        """
        import torch
        H = self.reasoning_hamiltonian
        diff = H - H.conj().T
        err = torch.norm(diff).item() / (torch.norm(H).item() + 1e-8)
        return (err < tol), err

    def evolve_non_hermitian_flow(
        self,
        state,
        dt: float = 0.1,
        gain_loss = None,
        nonreciprocal_coupling = None,
        renormalize: bool = True
    ):
        """
        Meaning 1 of Non-Hermitian Flow:
        Dynamical state evolution generated by a non-Hermitian Hamiltonian:
            H_eff = H_R - i * Gamma
        where H_R = (H + H^dagger)/2 is Hermitian (coherent oscillations),
        and Gamma is anti-Hermitian governing gain (amplification), loss (dissipation),
        and non-reciprocal transport (Arrow-of-Time directionality).

        Args:
            state: Complex state vector [..., dim]
            dt: Evolution timestep
            gain_loss: Diagonal or full gain/loss matrix [dim, dim]
            nonreciprocal_coupling: Asymmetric coupling matrix [dim, dim]
            renormalize: If True, re-normalizes state norm

        Returns:
            (evolved_state, effective_non_hermitian_hamiltonian)
        """
        import torch
        device = state.device
        dim = self.dim
        H_R = (self.reasoning_hamiltonian + self.reasoning_hamiltonian.conj().T) * 0.5

        if gain_loss is None:
            g_diag = harvest_honest_jitter((dim,), device=device, scaled=True) * 0.05
            Gamma = torch.diag(g_diag).to(torch.complex64)
        else:
            Gamma = gain_loss.to(device).to(torch.complex64)

        if nonreciprocal_coupling is not None:
            Gamma = Gamma + nonreciprocal_coupling.to(device).to(torch.complex64)

        H_eff = H_R - 1j * Gamma
        U = torch.eye(dim, dtype=torch.complex64, device=device) - 1j * H_eff * dt

        if state.dim() > 1:
            evolved = torch.matmul(state, U.T)
        else:
            evolved = torch.matmul(U, state)

        if renormalize:
            norm = torch.norm(evolved, dim=-1, keepdim=True)
            evolved = evolved / (norm + 1e-8)

        return evolved, H_eff

    def compute_spectral_flow(
        self,
        operator = None,
        parameter_trajectory = None,
        num_steps: int = 10
    ):
        """
        Meaning 2 of Non-Hermitian Flow:
        Tracks the motion of complex eigenvalues as parameters change (spectral flow):
            mu |-> {lambda_1(mu), ..., lambda_d(mu)} in C

        Returns:
            List of complex eigenvalue tensors [num_steps, dim]
        """
        import torch
        if operator is None:
            H_base = self.reasoning_hamiltonian
        else:
            H_base = operator

        device = H_base.device
        dim = H_base.shape[-1]

        if parameter_trajectory is None:
            parameter_trajectory = torch.linspace(0.0, 1.0, steps=num_steps, device=device)

        perturbation = torch.complex(
            harvest_honest_jitter((dim, dim), device=device, scaled=True) * 0.1,
            harvest_honest_jitter((dim, dim), device=device, scaled=True) * 0.1
        )

        flow_eigenvalues = []
        for mu in parameter_trajectory:
            H_mu = H_base + mu * perturbation
            eigs = torch.linalg.eigvals(H_mu)
            flow_eigenvalues.append(eigs)

        return flow_eigenvalues

    def detect_exceptional_points(
        self,
        operator = None,
        coalescence_threshold: float = 0.02
    ):
        """
        Meaning 2 Corollary: Detect proximity to an Exceptional Point (EP).
        An Exceptional Point occurs when two or more eigenvalues AND eigenvectors coalesce,
        rendering the Hamiltonian defective (non-diagonalizable Jordan block).

        Returns:
            (is_near_ep, min_eigenvalue_distance, eigenvector_condition_number)
        """
        import torch
        if operator is None:
            H = self.reasoning_hamiltonian
        else:
            H = operator

        try:
            evals, evecs = torch.linalg.eig(H)
        except Exception:
            return False, float('nan'), float('nan')

        dim = evals.shape[-1]
        if dim < 2:
            return False, 1.0, 1.0

        diffs = torch.abs(evals.unsqueeze(0) - evals.unsqueeze(1))
        diag_mask = torch.eye(dim, dtype=torch.bool, device=evals.device)
        diffs.masked_fill_(diag_mask, float('inf'))
        min_distance = torch.min(diffs).item()

        try:
            svals = torch.linalg.svdvals(evecs)
            cond = (torch.max(svals) / (torch.min(svals) + 1e-8)).item()
        except Exception:
            cond = float('inf')

        is_near_ep = (min_distance < coalescence_threshold) and (cond > 50.0)
        return is_near_ep, min_distance, cond
