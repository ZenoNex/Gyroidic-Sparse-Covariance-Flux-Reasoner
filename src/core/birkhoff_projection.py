"""
Stochastic Matrix Manifold (formerly Birkhoff Projection).

Role: Ensures that the DyadicTransferMap remains a valid stochastic matrix
(probabilistic flow constraints).

Uses the Sinkhorn-Knopp algorithm to project any transfer matrix T onto
the Birkhoff polytope (Doubly Stochastic) or Stochastic polytope.

Author: William Matthew Bryant
Created: January 2026
Refactored: January 2026 (Anti-Lobotomy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict, Any
import threading
from src.core.device_utils import DEVICE

# Global cache for pre-computed Birkhoff projectors to avoid redundant $O(N^4)$ re-init
_BIRKHOFF_PROJECTOR_CACHE: Dict[int, 'DirectBirkhoffProjection'] = {}
_BIRKHOFF_CACHE_LOCK = threading.Lock()
_BIRKHOFF_WARNED_DIMENSIONS = set()


class DModuleRankProbe(nn.Module):
    """
    D-Module Rank Probe.
    Replaces SparseRepunitProbe to prevent killing of high-entropy Voyenese.
    Uses NumericalDModuleManager to evaluate the true holonomic rank instead of geometric efficiency.
    """
    def __init__(self, state_dim: int, num_functionals: int):
        super().__init__()
        from src.core.numerical_d_module import NumericalDModuleManager
        self.d_module = NumericalDModuleManager(state_dim=state_dim, num_functionals=num_functionals)
        
    def forward(self, state: torch.Tensor, entropy: float = 0.5) -> Tuple[torch.Tensor, bool]:
        """
        If the state has sufficient D-Module rank (is not a Lazarus Void), it passes.
        Otherwise it is scaled down (repressed).
        Returns:
            processed_state: The preserved or repressed state
            is_feasible: True if rank was sufficient
        """
        batch_size = state.shape[0] if state.dim() > 1 else 1
        state_2d = state.view(batch_size, -1)
        
        # Approximate facets via self-outer product for rank evaluation
        num_funcs = min(state_2d.shape[-1], self.d_module.num_functionals)
        facets = torch.einsum('bi,bj->bij', state_2d, state_2d[:, :num_funcs])
        
        if num_funcs < self.d_module.num_functionals:
            pad = torch.zeros(batch_size, state_2d.shape[-1], self.d_module.num_functionals - num_funcs, device=state.device)
            facets = torch.cat([facets, pad], dim=-1)
            
        metrics = self.d_module(facets, entropy)
        is_feasible = not metrics["is_lazarus_void"]
        
        if is_feasible:
            return state, True
        else:
            return state * 0.1, False



class DirectBirkhoffProjection(nn.Module):
    """
    Sturmfels-Thomas Direct Linear Projection.
    
    Replaces O(N) iterative Sinkhorn with O(1) null-space projection
    onto the Birkhoff subspace H_Delta.
    
    Ensures that T satisfies:
        sum_j T_{ij} = 1
        sum_i T_{ij} = 1
    without iterative drift.
    """
    def __init__(self, n: int, device: str = None):
        super().__init__()
        self.n = n
        self.target_dim = n * n
        
        # Pre-compute the projection matrix onto the affine subspace Ax=b
        # where A encodes the row/column sum constraints.
        # This is high-memory for very large N, but O(1) at runtime.
        self._init_projection_matrix(device)

    def _init_projection_matrix(self, device):
        """Derive the projection matrix P derived from the constraint null-space."""
        n = self.n
        # Constraints: 2n linear equations (one redundant)
        # We build the constraint matrix A [2n, n*n]
        A = torch.zeros(2 * n, n * n, device=device)
        for i in range(n):
            # Row constraints
            A[i, i*n : (i+1)*n] = 1.0
            # Col constraints
            for j in range(n):
                A[n + j, i*n + j] = 1.0
        
        # Eliminate one redundant constraint to make A full rank (2n-1)
        A = A[:-1, :] 
        
        # P = I - A^T (A A^T)^-1 A
        # This projects any vector onto the nullspace of A (the subspace where sums are zero)
        # To project onto Ax=b, we use x_proj = Px + A^T(AA^T)^-1 b
        # Skillful De-allocation: Clear intermediate large matrices after computing P and p_offset
        # A_inv is pinverse of [2n-1, n*n]. For n=256, this is ~256M elements (1GB).
        A_inv = torch.pinverse(A)
        self.register_buffer('P', torch.eye(n*n, device=device) - torch.matmul(A_inv, A))
        
        # b is the target sums (all 1.0)
        b = torch.ones(2 * n - 1, 1, device=device)
        self.register_buffer('p_offset', torch.matmul(A_inv, b).squeeze())
        
        # Explicitly free memory
        del A
        del A_inv
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        Direct O(1) projection.
        T: [batch, n, n] or [batch, n*n]
        """
        shape = T.shape
        batch_size = shape[0]
        T_vec = T.view(batch_size, -1)
        
        # Linear map: x_proj = P x + offset
        T_proj = torch.matmul(T_vec, self.P.T) + self.p_offset
        
        return T_proj.view(shape)


class ObscuredBirkhoffManifold(nn.Module):
    """
    Obscured Birkhoff Polytope B_N^o.
    
    Fuses the Sturmfels direct projection with the evolved obstruction level.
    """
    
    def __init__(
        self, 
        n: int = 16, # Default manifold dimension
        temperature: float = 1.0,
        delta_o: float = 0.0,
        max_iterations: int = 50, # Added for compatibility
        device: str = None
    ):
        super().__init__()
        self.n = n
        # Clamp initial temperature to stable range
        self.temperature = nn.Parameter(torch.tensor(max(0.1, min(10.0, temperature))))
        self.register_buffer('delta_o', torch.tensor(delta_o))
        self.register_buffer('o', torch.zeros(n, device=device))
        self.max_iterations = max_iterations
        
        # Unicorn Synthesis: Use Direct Projection only for small dimensions to avoid OOM
        # A 256x256 matrix requires a 65536x65536 projection matrix (16GB).
        if n <= 32:
            with _BIRKHOFF_CACHE_LOCK:
                if n not in _BIRKHOFF_PROJECTOR_CACHE:
                    _BIRKHOFF_PROJECTOR_CACHE[n] = DirectBirkhoffProjection(n, device=device).to(DEVICE)
                self.direct_projector = _BIRKHOFF_PROJECTOR_CACHE[n]
        else:
            self.direct_projector = None
            with _BIRKHOFF_CACHE_LOCK:
                if n not in _BIRKHOFF_WARNED_DIMENSIONS:
                    _BIRKHOFF_WARNED_DIMENSIONS.add(n)
                    print(f" [CONFIG] Birkhoff dimension {n} too large for direct projection. Using iterative fallback.")
    
    def update_obscurity_from_state(self, state: torch.Tensor):
        """Update delta_o based on proximity to obstruction center o."""
        if hasattr(self, 'o') and self.o is not None:
            # Project state onto same dimension as o
            x = state.view(-1)
            if x.shape[0] >= self.o.shape[0]:
                x = x[:self.o.shape[0]]
            else:
                x = torch.nn.functional.pad(x, (0, self.o.shape[0] - x.shape[0]))
            dist = torch.norm(x - self.o)
            # Proximity-based visibility mask
            vis = torch.sigmoid(5.0 * dist)
            # Maximum dynamic obscurity is 0.5 (partial veiling) when state is close to o
            self.delta_o.copy_(0.5 * (1.0 - vis))

    def evolve_obstruction(self, genome: torch.Tensor, decay: float = 0.99):
        """delta_o = Obsc(g)"""
        target_obsc = torch.sigmoid(torch.mean(genome)) * 0.5
        self.delta_o = decay * self.delta_o + (1 - decay) * target_obsc
        
    def project(self, T: torch.Tensor, max_iterations: Optional[int] = None) -> torch.Tensor:
        """Project matrix T onto Birkhoff subspace with obstruction."""
        # 1. Log-domain Sinkhorn-Knopp for numerical stability
        temp = torch.clamp(self.temperature, min=0.01, max=10.0)
        log_T = T / temp
        
        iters = max_iterations if max_iterations is not None else self.max_iterations
        T_ds = self._sinkhorn_knopp_internal(log_T, iters)
        
        # 3. Apply Obstruction (delta_o)
        # sum_j T_ij = 1 - delta_o
        target_sum = 1.0 - self.delta_o
        return T_ds * target_sum

    def validate_stochasticity(self, T: torch.Tensor, tolerance: float = 1e-3) -> torch.Tensor:
        """
        Check if matrix T is on the Birkhoff polytope (doubly stochastic).
        
        Returns a boolean tensor of shape [batch, ...] indicating validity per sample.
        """
        row_sums = T.sum(dim=-1)
        col_sums = T.sum(dim=-2)
        
        target = 1.0 - self.delta_o
        
        # Compute max deviation from target sum
        row_err = (row_sums - target).abs().max(dim=-1)[0]
        col_err = (col_sums - target).abs().max(dim=-1)[0]
        
        return (row_err < tolerance) & (col_err < tolerance)

    def _sinkhorn_knopp_internal(self, log_T: torch.Tensor, iters: int) -> torch.Tensor:
        """
        Iterative Sinkhorn-Knopp algorithm in Log-Domain (Alternating Log-Softmax).
        Guarantees symmetrical row/column normalization without underflow.
        """
        for step in range(iters):
            # Row normalization (log-softmax)
            log_T = log_T - torch.logsumexp(log_T, dim=-1, keepdim=True)
            # Column normalization (log-softmax)
            log_T = log_T - torch.logsumexp(log_T, dim=-2, keepdim=True)
            
        return torch.exp(log_T)
    
    def forward(self, T: torch.Tensor, anneal: bool = False) -> torch.Tensor:
        """
        Forward pass with optional annealing.
        anneal is currently a placeholder for temperature scheduling.
        """
        return self.project(T)

BirkhoffProjection = ObscuredBirkhoffManifold



def sinkhorn_knopp(
    T: torch.Tensor, 
    max_iterations: int = 50, 
    temperature: float = 1.0,
    delta_o: float = 0.0
) -> torch.Tensor:
    """Functional wrapper for Sinkhorn-Knopp projection."""
    # Corrected constructor call
    manifold = ObscuredBirkhoffManifold(
        n=T.shape[-1], 
        max_iterations=max_iterations, 
        temperature=temperature,
        delta_o=delta_o
    )
    return manifold.project(T, max_iterations=max_iterations)

def project_to_birkhoff(T: torch.Tensor, max_iterations: int = 50) -> torch.Tensor:
    """Alternative name for sinkhorn_knopp with 2D shape fix."""
    # Ensure T is at least 3D for the manifold logic [batch, n, n]
    is_2d = T.dim() == 2
    if is_2d:
        T = T.unsqueeze(0)
        
    res = sinkhorn_knopp(T, max_iterations=max_iterations)
    
    if is_2d:
        return res.squeeze(0)
    return res

class BouligandBirkhoffProjectionFunction(torch.autograd.Function):
    """
    Custom Autograd Function implementing the projection onto the Birkhoff polytope
    with a Bouligand-correct gradient backpropagation.
    """
    @staticmethod
    def forward(ctx, T, manifold):
        ctx.manifold = manifold
        T_ds = manifold.project(T)
        ctx.save_for_backward(T_ds)
        return T_ds

    @staticmethod
    def backward(ctx, grad_output):
        T_ds, = ctx.saved_tensors
        manifold = ctx.manifold
        
        # T_ds_ij close to 0 indicates we are on the boundary
        eps = torch.finfo(T_ds.dtype).eps * 100.0
        boundary_mask = T_ds < max(eps, 1e-5)
        
        # Project incoming gradient onto the row/col sum zero-constraints
        shape = grad_output.shape
        grad_flat = grad_output.view(shape[0], -1)
        
        if getattr(manifold, 'direct_projector', None) is not None:
            grad_proj = torch.matmul(grad_flat, manifold.direct_projector.P.T)
            grad_proj = grad_proj.view(shape)
        else:
            grad_proj = grad_output - grad_output.mean(dim=-1, keepdim=True)
            grad_proj = grad_proj - grad_proj.mean(dim=-2, keepdim=True)
            
        # Apply Bouligand contingent cone projection
        grad_corrected = torch.where(boundary_mask & (grad_proj < 0), torch.zeros_like(grad_proj), grad_proj)
        
        num_corrected = (boundary_mask & (grad_proj < 0)).sum().item()
        if num_corrected > 0:
            print(f"[BOULIGAND_BIRKHOFF] Corrected backward gradients at {num_corrected} boundary constraints using B-derivative.", flush=True)
            
        return grad_corrected, None


class BouligandBirkhoffManifold(ObscuredBirkhoffManifold):
    """
    Obscured Birkhoff Polytope with Bouligand-correct gradient backpropagation.
    Inherits from ObscuredBirkhoffManifold for full backward compatibility.
    """
    def forward(self, T: torch.Tensor, anneal: bool = False) -> torch.Tensor:
        return BouligandBirkhoffProjectionFunction.apply(T, self)


# Post-import to avoid circular dependency, exposing the hybrid probe to System 2 Constraint Operator
from .cayley_cubic_probe import CayleyCubicProbe
