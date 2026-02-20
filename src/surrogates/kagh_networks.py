"""
KAGH-Boltzmann Networks: Kolmogorov-Arnold-Gödel-Huxley-Boltzmann.

Hybrid geometric architecture for Physics-ADMM surrogates.
Components:
    - KAN: Kolmogorov-Arnold Networks (B-spline basis)
    - HuxleyRD: Reaction-Diffusion dynamics for biological sparsity
    - Gödel: Adaptive soft-logic gates
    - Boltzmann: Stochastic admissibility sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Union, Dict, Callable

class SaturatedQuantizer(torch.autograd.Function):
    """
    Hybrid Quantization Primitive.
    
    Acts as the "Inter-Domain" bridge between the Continuous (Physics) and 
    Discrete (Symbolic) regimes.
    
    Forward: Snaps weights to 'levels' discrete steps (Saturated Reserves).
    Backward: Straight-Through Estimator (STE) allows gradient flow to find 
              the best discrete configuration.
    """
    @staticmethod
    def forward(ctx, input, levels=64):
        # Scale to integer grid [-levels/2, levels/2]
        scale = levels / 2.0
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient through unchanged
        return grad_output, None

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer (True B-Spline Implementation).
    
    Features:
    - True B-Spline Basis (Cox-de Boor recursion).
    - Hybrid-Quantized Weights (SaturatedQuantizer).
    - Fixed Structural Grids (Non-Teleological).
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, quantization_levels: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.quantization_levels = quantization_levels
        
        # Base weight (linear residual) - "Silu" activation often used in KAN papers, 
        # but here we keep linear base + spline non-linearity
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline weights: [out, in, grid_size + spline_order]
        self.spline_weight = nn.Parameter(torch.Tensor(
            out_features, in_features, grid_size + spline_order
        ))
        
        # Fixed Structural Grid
        # "Symbolic Non-Revisability": The grid is an immutable topological reference.
        # Range is typically normalized [-1, 1] for KANs
        h = 2.0 / grid_size # range size 2 / grid_size
        # Extended grid for B-splines: need k points padding on both sides
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h - 1.0
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid) # [in, G+2k+1]
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        # Init spline weights near zero (small perturbations)
        # Using a scaled noise to encourage finding structure
        nn.init.normal_(self.spline_weight, 0.0, 0.1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute True B-spline basis functions using Vectorized Cox-de Boor.
        
        x: [batch, in] (assumed normalized [-1, 1])
        Returns: [batch, in, grid_size + spline_order]
        """
        x = x.unsqueeze(-1) # [batch, in, 1]
        grid = self.grid    # [in, grid_len]
        
        k = self.spline_order
        
        # Level 0: piecewise constant (1 if x in [t_i, t_{i+1}), 0 otherwise)
        # We compute this for all relevant intervals
        # The number of basis functions at order k is N = G + k
        # We need to start with N + k intervals (roughly)
        
        # Efficient Vectorized Recursion
        # B_{i, 0}(x)
        # We consider intervals [t_i, t_{i+1})
        # i ranges from 0 to grid_len - 2
        
        # Extend grid dimensions for broadcasting: [1, in, grid_len]
        grid_broad = grid.unsqueeze(0)
        
        # Logic: 1 if grid[i] <= x < grid[i+1]
        # We take the first G + 2k basis functions of order 0
        # (Actually, we just need to recurse up to order k)
        
        # Implementation note: It's often easier to compute *all* B_i,0 and reduce
        bases = ((x >= grid_broad[:, :, :-1]) & (x < grid_broad[:, :, 1:])).float()
        
        # Recursion: B_{i, p}(x)
        for p in range(1, k + 1):
            # We compute B_{i, p} from B_{i, p-1} and B_{i+1, p-1}
            # The number of functions decreases by 1 at each step
            
            # Left term: (x - t_i) / (t_{i+p} - t_i) * B_{i, p-1}
            t_i = grid_broad[:, :, :-1-p] # Shifts for current index
            t_i_p = grid_broad[:, :, p:-1] # t_{i+p}
            
            numer1 = x - t_i
            denom1 = t_i_p - t_i + 1e-8
            term1 = (numer1 / denom1) * bases[:, :, :-1]
            
            # Right term: (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1}) * B_{i+1, p-1}
            t_i_1 = grid_broad[:, :, 1:-p] # t_{i+1}
            t_i_p_1 = grid_broad[:, :, p+1:] # t_{i+p+1}
            
            numer2 = t_i_p_1 - x
            denom2 = t_i_p_1 - t_i_1 + 1e-8
            term2 = (numer2 / denom2) * bases[:, :, 1:]
            
            bases = term1 + term2
            
        # bases is now [batch, in, G+k] (approximately, dependent on grid setup)
        # We need to ensure we return exactly self.spline_weight.shape[-1] columns
        target_dim = self.spline_weight.shape[-1]
        if bases.shape[-1] > target_dim:
            bases = bases[:, :, :target_dim]
        elif bases.shape[-1] < target_dim:
             # This shouldn't happen with correct grid sizing, but detailed padded required
             padding = torch.zeros(bases.shape[0], bases.shape[1], target_dim - bases.shape[-1], device=x.device)
             bases = torch.cat([bases, padding], dim=-1)
             
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear structural residual
        base_output = F.linear(x, self.base_weight)
        
        # B-Spline activation
        # Assume x is roughly in [-1, 1] or normalize. 
        # KANs usually require bounded input or grid adaptation. 
        # With fixed grid, we rely on upstream normalization (e.g. Tanh or LayerNorm)
        # The architecture uses Tanh on embedding, so inputs are bounded.
        
        # Evaluate basis
        basis = self.b_splines(x) # [batch, in, coeff_dim]
        
        # Hybrid Quantization of Spline Weights
        # "Inter-Domain Contract": Weights must be quantized
        q_weight = SaturatedQuantizer.apply(self.spline_weight, self.quantization_levels)
        
        # Linear combination: y = sum(w_i * b_i(x))
        spline_output = torch.einsum('bic,oic->bo', basis, q_weight)
        
        return base_output + spline_output

class HarmonicWaveDecomposition(nn.Module):
    """
    Decomposes input into Ergodic and Non-Ergodic spectral components.
    
    "Let part of the harmonic wave decomposition... carry through non-ergodic sub-dynamics"
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Learnable spectral gate
        self.spectral_gate = nn.Parameter(torch.ones(dim // 2 + 1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (ergodic_component, non_ergodic_component)
        """
        # 1. Harmonic Decomposition (RFFT)
        # Treat features as spatial signal
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 2. Spectral Gating
        # High freq -> Non-Ergodic (Solitons)
        # Low freq -> Ergodic (Mixing)
        
        # Simple split or soft gate? Soft gate for differentiability
        gate = torch.sigmoid(self.spectral_gate)
        
        ergodic_freq = x_freq * gate
        non_ergodic_freq = x_freq * (1.0 - gate)
        
        ergodic_part = torch.fft.irfft(ergodic_freq, n=self.dim, dim=-1)
        non_ergodic_part = torch.fft.irfft(non_ergodic_freq, n=self.dim, dim=-1)
        
        return ergodic_part, non_ergodic_part

class TrigonometricUnfolding(nn.Module):
    """
    Speculative Primitive: Trigonometric Gyroid Unfolding Operator.
    
    Handles "casus irreducibilis" when polynomial bases become degenerate.
    Reveals hidden negentropic solitons via triple-angle unfolding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.tau_decay = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, u_h: torch.Tensor, gcve_pressure: torch.Tensor, chirality: torch.Tensor) -> torch.Tensor:
        """
        u_h: [batch, features] non-ergodic component
        gcve_pressure: [batch] Gyroid violation
        chirality: [batch] Global chirality index
        """
        if gcve_pressure is None:
            return u_h
            
        # 1. Compute Phase Parameter phi
        # cos(3phi) = [3V - tr(C)/tau] / [2 * (-det(PAS))^1.5]
        # Approximate stats from u_h
        tr_c = torch.sum(u_h**2, dim=-1) # Local covariance trace proxy
        u_h_freq = torch.fft.rfft(u_h, dim=-1)
        det_pas_proxy = torch.var(torch.abs(u_h_freq), dim=-1) + 1e-6 # Spectral variance
        
        denominator = 2 * (torch.sqrt(det_pas_proxy))**3 + 1e-8
        cos_3phi = (3 * gcve_pressure - tr_c / self.tau_decay) / denominator
        cos_3phi = torch.clamp(cos_3phi, -0.999, 0.999) # Ensure stability
        phi = torch.acos(cos_3phi) / 3.0
        
        # 2. Unfold into 3 branches (k=0, 1, 2)
        # u_h^(k) = 2 * sqrt(-lambda_min/3) * cos(phi + 2pi*k/3)
        lambda_min_proxy = torch.min(torch.abs(u_h_freq), dim=-1)[0]
        
        branches = []
        for k in [0, 1, 2]:
            amp = 2 * torch.sqrt(lambda_min_proxy / 3.0 + 1e-8)
            cos_term = torch.cos(phi + 2 * math.pi * k / 3.0)
            # Chiral shift: k * chi (Simplified interaction)
            shift = torch.exp(-torch.abs(chirality) * k) if chirality is not None else 1.0
            branches.append(amp.unsqueeze(-1) * cos_term.unsqueeze(-1) * u_h * shift.unsqueeze(-1))
            
        # 3. Negentropic Branch Selection
        # Select k that maximizes negentropy (proxied by energy in the logic structure)
        energies = torch.stack([torch.norm(b, dim=-1) for b in branches]) # [3, batch]
        best_k = torch.argmax(energies, dim=0) # [batch]
        
        u_h_unfolded = torch.zeros_like(u_h)
        for i, branch in enumerate(branches):
            mask = (best_k == i).unsqueeze(-1)
            u_h_unfolded += branch * mask
            
        return u_h_unfolded

class HuxleyRD(nn.Module):
    """
    Huxley Reaction-Diffusion Layer with Non-Ergodic Sub-Dynamics.
    
    Splits signal into:
    1. Diffusive (Ergodic) component: Mixes via diffusion kernel.
    2. Soliton (Non-Ergodic) component: Propagates without spreading.
    """
    def __init__(self, num_features: int, tau: float = 0.1):
        super().__init__()
        self.tau = tau
        # Learnable diffusion coupling
        self.diffusion_kernel = nn.Parameter(torch.tensor([0.1, -0.2, 0.1]).view(1, 1, 3))
        self.a = nn.Parameter(torch.tensor(0.1)) # Threshold parameter
        self.b = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        # Wave Decomposition
        self.wave_decomp = HarmonicWaveDecomposition(num_features)
        
        # Phase shift for non-ergodic component (Soliton velocity)
        self.soliton_velocity = nn.Parameter(torch.tensor(0.0))
        
        # Trigonometric Unfolding (Geometric Revelation)
        self.unfolding = TrigonometricUnfolding(num_features)

    def forward(self, u: torch.Tensor, gcve_pressure: Optional[torch.Tensor] = None, chirality: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u: [batch, features] representing coefficients on a 1D grid
        """
        # 1. Decompose
        u_ergodic, u_non_ergodic = self.wave_decomp(u)
        
        # --- Ergodic Dynamics (Reaction-Diffusion) ---
        # Reaction: du/dt = u(u - a)(1 - u)
        reaction = u_ergodic * (u_ergodic - self.a) * (1.0 - u_ergodic)
        
        # Diffusion: 1D convolution
        u_reshaped = u_ergodic.unsqueeze(1)
        diffusion = F.conv1d(u_reshaped, self.diffusion_kernel, padding=1)
        diffusion = diffusion.squeeze(1)
        
        du_ergodic = reaction + self.gamma * diffusion
        u_ergodic_next = u_ergodic + self.tau * du_ergodic
        
        # --- Non-Ergodic Dynamics (Soliton / Phase Shift) ---
        # Apply pure phase shift
        u_ne_freq = torch.fft.rfft(u_non_ergodic, dim=-1)
        k_indices = torch.arange(u_ne_freq.shape[-1], device=u.device)
        phase_shift = torch.exp(-1j * 2 * math.pi * k_indices * self.soliton_velocity / u.shape[-1])
        u_ne_shifted = torch.fft.irfft(u_ne_freq * phase_shift, n=u.shape[-1], dim=-1)
        
        # --- Geometric Revelation: Trigonometric Unfolding ---
        # When violations are high, "unfold" the soliton channel to avoid collapse
        u_ne_unfolded = self.unfolding(u_ne_shifted, gcve_pressure, chirality)
        
        # Recombine
        # "Ergodic Soliton Fusion Gate" (Speculative Primitive)
        # Fusion Gate
        fusion = F.sigmoid(u_ergodic_next + u_ne_unfolded)
        
        return fusion * (u_ergodic_next + u_ne_unfolded)

def goedel_positivity(x: torch.Tensor, epsilon: float = 1e-6, active: bool = True) -> torch.Tensor:
    """
    Gödel Logic Gate: Soft differentiable constraint satisfaction.
    Enforces x >= 0 via: x <- x * sigmoid(k * (x - epsilon))
    
    If active=False (Symbolic regime or inference), this is a no-op to allow signed residues.
    """
    if not active:
        return x
    k = 100.0
    gate = torch.sigmoid(k * (x - epsilon))
    return x * gate

class KAGHBlock(nn.Module):
    """
    Full KAGH-Boltzmann Admissibility Block.
    """
    def __init__(self, n_in: int, n_out: int, width: int = 64, depth: int = 3, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        
        # KAN Skeletal Layering
        layers = []
        in_dim = n_in
        for i in range(depth):
            out_dim = n_out if i == depth - 1 else width
            layers.append(KANLayer(in_dim, out_dim))
            in_dim = width
        self.kan_layers = nn.ModuleList(layers)
        
        # Dynamics
        self.diffusion = HuxleyRD(n_out)
        
        # Boltzmann Temperature (learnable)
        self.log_temp = nn.Parameter(torch.tensor(0.0)) # exp(0) = 1.0
        
        # Structural Blindness: Freeze topology
        self.is_fossilized = False
        
    def fossilize(self):
        """Freeze KAN skeletal layers to prevent logical emulation."""
        self.is_fossilized = True
        for param in self.kan_layers.parameters():
            param.requires_grad = False

    def forward(self, c: torch.Tensor, M_mv: callable = None, use_boltzmann: bool = True, gcve_pressure: Optional[torch.Tensor] = None, chirality: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            c: [batch, n_in] input coefficients
            M_mv: Optional anisotropic operator (M^alpha)
            gcve_pressure: Optional topological violation
            chirality: Optional identity preservation index
        """
        x = c
        
        # 1. KAN Flow
        for lay in self.kan_layers:
            x = lay(x)
            
        # 2. Huxley Reaction-Diffusion (Sparsity/Pattern Formation) + Unfolding
        x = self.diffusion(x, gcve_pressure=gcve_pressure, chirality=chirality)
        
        # 3. Gödel Gate (Control)
        # Disable Gödel gates during repair/inference to prevent bias
        is_inferring = (gcve_pressure is not None) 
        x = goedel_positivity(x, active=(not is_inferring and self.training)) 
        
        # 4. Boltzmann Sampling (Stochasticity only at boundaries/adaptation)
        if use_boltzmann and self.training and not is_inferring:
            temp = torch.exp(self.log_temp)
            noise = torch.randn_like(x) * torch.sqrt(temp)
            x = x + noise
            
        return x
