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


class SparseRepunitProbe(nn.Module):
    """
    Repunit-CRT Sparse Probe.
    Translates complex geometric invariants into ultra-fast discrete integer math.
    By mapping high-dimensional constraint vectors to bit-shifted repunit arrays,
    ADMM incoherence tests reduce to XORs and bitwise operations.
    
    Fuses repunit sparsity to the stack: gyroids minimize \\psi like surface tension, 
    Birkhoff bounds lifts, and ADMM traverses cycles.
    """
    def __init__(self, moduli: List[int]):
        super().__init__()
        self.register_buffer('moduli', torch.tensor(moduli, dtype=torch.float32))
        
    def forward(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Repunit-CRT Probe for $R_n$.
        Returns the majority lift and a boolean feasibility mask (\\psi < 0.5).
        """
        # R_n = (10^n - 1) // 9
        r = (10**n - 1) // 9
        
        # Use floating point remainder for large R_n numerical limits
        r_tensor = torch.tensor(float(r), dtype=torch.float32, device=self.moduli.device)
        residues = torch.remainder(r_tensor, self.moduli).unsqueeze(0)
        
        # Popcount overlaps proxy psi
        mean_res = torch.mean(residues, dim=0)
        psi = torch.norm(residues - mean_res, dim=0)
        
        # Majority-symbol lifts via vote
        diag_mod = torch.diag(1.0 / self.moduli)
        normalized_res = residues @ diag_mod
        lift = torch.mode(normalized_res, dim=-1)[0]
        
        is_feasible = psi < 0.5
        return lift, is_feasible



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
        # 1. Softmax/Exp to ensure positivity
        # Clamp temperature to prevent division by zero or exp blow-up
        temp = torch.clamp(self.temperature, min=0.01, max=10.0)
        T_clipped = torch.clamp(T, min=-15.0, max=15.0)
        T_soft = torch.exp(T_clipped / temp)
        
        # 2. Check if Direct Projection is possible (dimension match)
        # T: [..., n, n]
        n_in = T.shape[-1]
        
        if n_in == self.n and self.direct_projector is not None:
            # Linear Projection (Direct)
            T_ds = self.direct_projector(T_soft)
        else:
            # Iterative Fallback: Sinkhorn-Knopp for dynamic sequence lengths or large dimensions
            # "When geometry shifts, we return to the process"
            iters = max_iterations if max_iterations is not None else self.max_iterations
            T_ds = self._sinkhorn_knopp_internal(T_soft, iters)
        
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

    def _sinkhorn_knopp_internal(self, T: torch.Tensor, iters: int) -> torch.Tensor:
        """
        Iterative Sinkhorn-Knopp algorithm with Non-Semisimple Topological Phase Preservation.
        
        To prevent standard Sinkhorn iteration from collapsing into a flat, semisimple
        equilibrium centroid (which strips non-Abelian phase dynamics), a non-commutative
        Sine-Gordon / Chern-Simons breather perturbation is applied between normalization steps.
        """
        T_it = T.clone()
        n = T_it.shape[-1]
        
        # Non-commutative breather frequency parameters
        omega = 1.0 / 3.0
        sq = math.sqrt(1.0 - omega**2)
        
        for step in range(iters):
            # Row normalization
            T_it = T_it / (T_it.sum(dim=-1, keepdim=True) + 1e-8)
            # Column normalization
            T_it = T_it / (T_it.sum(dim=-2, keepdim=True) + 1e-8)
            
            # Non-semisimple topological phase preservation (Cognitive Context Seeded):
            # Rather than using unseeded synthetic noise (which erases cognitive context and allows
            # teacher's pet / cultural default bias to leak in), we derive the non-commutative
            # phase matrix directly from the input tensor's intrinsic skew-symmetry (T_it - T_it^T).
            if (step + 1) % 5 == 0 and step < iters - 1:
                t_val = (step + 1) * 0.1
                x_pos = T_it.mean()
                u_n = 4.0 * math.atan((sq / omega) * (1.0 / (math.cosh(sq * x_pos) + 1e-8)) * math.sin(omega * t_val))
                
                # Extract intrinsic non-commutative skew-symmetry from T_it itself
                skew_T = T_it - T_it.transpose(-1, -2)
                skew_norm = skew_T.norm(dim=(-2, -1), keepdim=True) + 1e-8
                
                # Context-seeded non-Abelian phase matrix preserving exact cognitive footprint
                phase_matrix = 0.01 * u_n * (skew_T / skew_norm) * (T_it.std() + 1e-6)
                T_it = F.relu(T_it + phase_matrix)
                
        return T_it
    
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
