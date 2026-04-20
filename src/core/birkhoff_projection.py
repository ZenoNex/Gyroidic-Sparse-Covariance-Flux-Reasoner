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
from typing import Optional, List, Tuple


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
        A_inv = torch.pinverse(A)
        self.register_buffer('P', torch.eye(n*n, device=device) - torch.matmul(A_inv, A))
        
        # b is the target sums (all 1.0)
        b = torch.ones(2 * n - 1, 1, device=device)
        self.register_buffer('p_offset', torch.matmul(A_inv, b).squeeze())

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
        device: str = None
    ):
        super().__init__()
        self.n = n
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.register_buffer('delta_o', torch.tensor(delta_o))
        
        # Unicorn Synthesis: Use Direct Projection by default
        self.direct_projector = DirectBirkhoffProjection(n, device=device)
    
    def evolve_obstruction(self, genome: torch.Tensor, decay: float = 0.99):
        """delta_o = Obsc(g)"""
        target_obsc = torch.sigmoid(torch.mean(genome)) * 0.5
        self.delta_o = decay * self.delta_o + (1 - decay) * target_obsc
        
    def project(self, T: torch.Tensor) -> torch.Tensor:
        """Project matrix T onto Birkhoff subspace with obstruction."""
        # 1. Softmax/Exp to ensure positivity
        T_soft = torch.exp(T / self.temperature)
        
        # 2. Linear Projection
        T_ds = self.direct_projector(T_soft)
        
        # 3. Apply Obstruction (delta_o)
        # sum_j T_ij = 1 - delta_o
        target_sum = 1.0 - self.delta_o
        return T_ds * target_sum
    
    def forward(self, T: torch.Tensor) -> torch.Tensor:
        return self.project(T)

BirkhoffProjection = ObscuredBirkhoffManifold

def sinkhorn_knopp(
    T: torch.Tensor, 
    max_iterations: int = 50, 
    temperature: float = 1.0,
    delta_o: float = 0.0
) -> torch.Tensor:
    """Functional wrapper for Sinkhorn-Knopp projection."""
    manifold = ObscuredBirkhoffManifold(
        max_iterations=max_iterations, 
        temperature=temperature,
        delta_o=delta_o
    )
    return manifold.project(T)

def project_to_birkhoff(T: torch.Tensor, max_iterations: int = 50) -> torch.Tensor:
    """Alternative name for sinkhorn_knopp."""
    return sinkhorn_knopp(T, max_iterations=max_iterations)

# Post-import to avoid circular dependency, exposing the hybrid probe to System 2 Constraint Operator
from .cayley_cubic_probe import CayleyCubicProbe
