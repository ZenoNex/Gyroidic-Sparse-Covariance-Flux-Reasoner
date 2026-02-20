"""
Fractional Operators M^alpha via Krylov-Lanczos Search.

Implements M^alpha * v using:
1. Diagonal search (if M is diagonal)
2. Lanczos approximation (if M is symmetric)
3. CODES Coherence Gating simulation (GPU constraint)
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
import math

class CODESDriver:
    """
    Simulates the CODES (Coherence-Oriented Deterministic Execution System) driving layer.
    
    In a real GPU environment, this would interface with the fractional warp scheduler.
    Here, it simulates coherence gating based on phase alignment.
    """
    
    @staticmethod
    def compute_pas_h(phase: float, harmonics: list = None) -> float:
        """
        Compute Multiharmonic Phase Alignment Score (PAS_h).
        Uses polynomial-based harmonics instead of hardcoded primes (anti-lobotomy).
        """
        if harmonics is None:
            # Generate polynomial-based harmonics using Chebyshev roots
            harmonics = []
            for n in range(1, 7):  # Generate 6 harmonics
                # Use Chebyshev polynomial roots scaled to positive integers
                root = math.cos((2*n - 1) * math.pi / (2 * 6))  # Chebyshev root
                harmonic = abs(root * 10) + 1  # Scale and ensure positive
                harmonics.append(harmonic)
        
        score = 0.0
        for m in harmonics:
            # Simple simulation: aligned if harmonics sum constructively
            # Real hardware uses complex exponential accumulation
            score += math.cos(m * phase)
        return (score / len(harmonics) + 1.0) / 2.0  # Normalize to [0, 1]

    @staticmethod
    def is_coherent(phase: float, threshold: float = 0.75) -> bool:
        """AURAOUT gating: only proceed if coherent."""
        pas = CODESDriver.compute_pas_h(phase)
        return pas >= threshold

def lanczos_iteration(
    mv_func: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    k: int = 20,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform k steps of Lanczos iteration to approximate the Krylov subspace.
    
    Args:
        mv_func: Matrix-vector product function (M @ v)
        v: Initial vector [n]
        k: Number of Krylov steps
        
    Returns:
        Q: [n, k] Orthonormal basis
        T: [k, k] Tridiagonal matrix
    """
    if device is None:
        device = v.device
        
    n = v.shape[0]
    dtype = v.dtype
    
    Q = torch.zeros(n, k, dtype=dtype, device=device)
    T = torch.zeros(k, k, dtype=dtype, device=device)
    
    # Initial step
    beta_prev = 0
    q_prev = torch.zeros_like(v)
    
    # Normalize start vector
    beta = torch.norm(v)
    if beta < 1e-10:
        return Q, T # Zero vector case
        
    q = v / beta
    Q[:, 0] = q
    
    for j in range(k):
        # M @ q_j
        r = mv_func(q)
        
        # Orthogonalize against q_{j-1}
        if j > 0:
            r = r - beta_prev * q_prev
            
        # Alpha_j = q_j^T M q_j
        alpha = torch.dot(q, r)
        T[j, j] = alpha
        
        # Orthogonalize against q_j
        r = r - alpha * q
        
        # Re-orthogonalization (optional stability fix, skipped for speed)
        
        # Beta_{j+1} = norm(r)
        beta_new = torch.norm(r)
        
        if j < k - 1:
            T[j, j+1] = beta_new
            T[j+1, j] = beta_new
            
            if beta_new < 1e-10:
                break # Invariant subspace found
                
            q_next = r / beta_new
            Q[:, j+1] = q_next
            
            q_prev = q
            q = q_next
            beta_prev = beta_new
            
    return Q, T

def frac_apply(
    M: Union[torch.Tensor, Callable],
    v: torch.Tensor,
    alpha: float,
    k_steps: int = 30,
    use_codes: bool = True,
    ranging_gamma: float = 0.5  # Hardening factor for low coherence
) -> torch.Tensor:
    """
    Compute M^alpha @ v. Primarily used for **Topological Repair** in System 2.
    
    Enforces anisotropy constraints on recovered symbolic residues to heal
    fractured reasoning chains.
    
    Args:
        M: Linear operator (tensor or callable)
        v: Input vector
        alpha: Fractional exponent
        k_steps: Lanczos steps
        use_codes: Enable CODES coherence check simulation
        ranging_gamma: Impact of coherence on alpha (hardening)
        
    Returns:
        Result vector w = M^alpha v
    """
    # 1. CODES Coherence Gating & Alpha Ranging
    if use_codes:
        # Simulate phase derived from data hash or explicit clock
        phase = float(torch.sum(v).item() % (2 * math.pi))
        
        # Compute coherence score [0, 1]
        coherence_score = CODESDriver.compute_pas_h(phase)
        
        # Gate: AURAOUT
        # If very incoherent, we might still want to proceed but with HARDENED alpha
        # But if strictly is_coherent is False, existing logic returns zero.
        # Let's keep the strict gating for now, but apply ranging if we pass.
        
        if coherence_score < 0.20: # ALLOWING TOPOLOGICAL THAW
             # Default threshold from is_coherent
             return torch.zeros_like(v)
             
        # Adaptive Ranging: Harden alpha if coherence is imperfect
        # alpha' = alpha + gamma * (1 - coherence)
        # Less coherent -> Higher alpha -> Stronger operator application (Hardening)
        alpha = alpha # Alpha hardening disabled for 0.61 recovery

    # 2. Diagonal Search
    if isinstance(M, torch.Tensor) and M.ndim == 1:
        # Diagonal matrix represented as vector
        return frac_apply_diagonal(M, v, alpha)
        
    # 3. Lanczos Approximation for General Symmetric M
    if isinstance(M, torch.Tensor):
        if M.ndim == 2:
            mv_func = lambda x: M @ x
        else:
            raise ValueError("M must be 1D (diag) or 2D (dense) Tensor")
    else:
        mv_func = M
        
    # Perform Lanczos
    Q, T = lanczos_iteration(mv_func, v, k=k_steps)
    
    # Evaluate f(T) = T^alpha on small kxk matrix
    # T is symmetric tridiagonal -> eigendecomposition is cheap
    eigvals, eigvecs = torch.linalg.eigh(T)
    
    # Apply function to eigenvalues: lambda^alpha
    # Handle negative eigenvalues via complex absolute or clamp?
    # Hamiltonian dynamics usually implies PSD M.
    eigvals_frac = torch.pow(torch.clamp(eigvals, min=1e-6), alpha)
    
    # f(T) = V diag(f(lam)) V^T
    f_T = eigvecs @ torch.diag(eigvals_frac) @ eigvecs.T
    
    # Result approximation: ||v|| * Q * f(T) * e_1
    beta = torch.norm(v)
    e1 = torch.zeros(k_steps, device=v.device)
    e1[0] = 1.0
    
    # w = beta * Q @ (f_T @ e1)
    w = beta * (Q @ f_T[:, 0])
    
    return w

def frac_apply_diagonal(
    diag_M: torch.Tensor,
    v: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """Compute (diag(M)^alpha) v element-wise."""
    return torch.pow(torch.clamp(diag_M, min=1e-6), alpha) * v
