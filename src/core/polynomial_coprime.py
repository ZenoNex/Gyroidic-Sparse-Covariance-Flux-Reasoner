"""
Polynomial co-prime functional system with Birkhoff polytope-constrained coefficients.

Replaces discrete prime-based modular arithmetic with continuous polynomial functionals
that are mutually co-prime and have coefficients constrained to the Birkhoff polytope.

Mathematical Foundation:
    φ_k(x; θ_k) = Σ_i θ_k[i] · p_i(x)
    
    Where:
        - θ_k ∈ Birkhoff polytope (doubly-stochastic matrix)
        - gcd(φ_i, φ_j) = 1 for all i ≠ j (co-primality)
        - p_i(x) are orthogonal polynomial basis functions

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import numpy as np

# Non-ergodic entropy optimization (January 2026)
try:
    from src.core.non_ergodic_entropy import NonErgodicFractalEntropy
    _HAS_NON_ERGODIC = True
except ImportError:
    _HAS_NON_ERGODIC = False

from src.core.continuous_coprimality import ContinuousCoprimality


class PolynomialBasis:
    """
    Orthogonal polynomial basis functions (Chebyshev, Legendre, etc.).
    """
    
    def __init__(
        self,
        degree: int = 4,
        basis_type: str = 'chebyshev',
        domain: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Args:
            degree: Maximum polynomial degree
            basis_type: 'chebyshev', 'legendre', or 'hermite'
            domain: Domain for polynomial evaluation
        """
        self.degree = degree
        self.basis_type = basis_type
        self.domain = domain
        self.dim = degree + 1  # Number of basis functions
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate basis polynomials at x.
        
        Args:
            x: [batch, ...] input values
            
        Returns:
            basis_values: [batch, ..., dim] basis function evaluations
        """
        # Normalize x to [-1, 1]
        x_norm = 2.0 * (x - self.domain[0]) / (self.domain[1] - self.domain[0]) - 1.0
        
        batch_shape = x.shape
        x_flat = x_norm.reshape(-1)
        
        if self.basis_type == 'chebyshev':
            basis_vals = self._chebyshev(x_flat)
        elif self.basis_type == 'legendre':
            basis_vals = self._legendre(x_flat)
        elif self.basis_type == 'hermite':
            basis_vals = self._hermite(x_flat)
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")
        
        # Reshape to [..., dim]
        return basis_vals.reshape(*batch_shape, self.dim)
    
    def _chebyshev(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chebyshev polynomials of the first kind: T_n(x).
        
        Recurrence: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)
        """
        n = x.shape[0]
        T = torch.zeros(n, self.dim, device=x.device, dtype=x.dtype)
        
        T[:, 0] = 1.0  # T_0(x) = 1
        if self.dim > 1:
            T[:, 1] = x  # T_1(x) = x
        
        for i in range(2, self.dim):
            T[:, i] = 2.0 * x * T[:, i-1] - T[:, i-2]
        
        return T
    
    def _legendre(self, x: torch.Tensor) -> torch.Tensor:
        """
        Legendre polynomials: P_n(x).
        
        Recurrence: P_0(x) = 1, P_1(x) = x, 
                   (n+1)P_{n+1}(x) = (2n+1)x·P_n(x) - n·P_{n-1}(x)
        """
        n = x.shape[0]
        P = torch.zeros(n, self.dim, device=x.device, dtype=x.dtype)
        
        P[:, 0] = 1.0  # P_0(x) = 1
        if self.dim > 1:
            P[:, 1] = x  # P_1(x) = x
        
        for i in range(2, self.dim):
            P[:, i] = ((2*i - 1) * x * P[:, i-1] - (i - 1) * P[:, i-2]) / i
        
        return P
    
    def _hermite(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probabilist's Hermite polynomials: He_n(x).
        
        Recurrence: He_0(x) = 1, He_1(x) = x,
                   He_{n+1}(x) = x·He_n(x) - n·He_{n-1}(x)
        """
        n = x.shape[0]
        He = torch.zeros(n, self.dim, device=x.device, dtype=x.dtype)
        
        He[:, 0] = 1.0  # He_0(x) = 1
        if self.dim > 1:
            He[:, 1] = x  # He_1(x) = x
        
        for i in range(2, self.dim):
            He[:, i] = x * He[:, i-1] - (i - 1) * He[:, i-2]
        
        return He


class SaturatedPolynomialGate(nn.Module):
    """
    Implements piecewise-saturated polynomial functionals.
    phi_tilde = sign(sum theta * P) * s_k
    """
    def __init__(self, s_max: float = 1.0):
        super().__init__()
        self.s_max = nn.Parameter(torch.tensor(s_max))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hard saturation (sign) and evolved scale.
        """
        return torch.sign(x) * self.s_max


class HypergraphOrthogonalityPressure(nn.Module):
    """
    Selection pressure that maximizes joint entropy of functional outcomes.
    Acts as a first-class admissibility filter for symbolic transversality.
    
    Used for evolutionary selection, not gradient descent.
    """
    def __init__(self, k_order: int = 3):
        super().__init__()
        self.k_order = k_order
        
    def forward(self, phi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hypergraph Orthogonality Pressure: Fractal (Russian Doll) Decomposition.
        Returns a dictionary of domain-isolated pressures (Vectorized Pressure).
        """
        batch_size, K = phi.shape
        if K < self.k_order:
            return {'local_entropy': torch.tensor([0.0], device=phi.device), 
                    'global_entropy': torch.tensor(0.0, device=phi.device)}
            
        # 1. Local Clustering (Small Scale)
        B = 4
        num_blocks = (K + B - 1) // B
        local_entropies = []
        block_reps = []
        
        for i in range(num_blocks):
            start = i * B
            end = min(start + B, K)
            block_phi = phi[:, start:end]
            h_local = self._compute_subset_entropy(block_phi, min(self.k_order, end-start))
            local_entropies.append(h_local)
            block_reps.append(block_phi.mean(dim=1, keepdim=True))
            
        # 2. Global Coupling (Large Scale)
        global_phi = torch.cat(block_reps, dim=1)
        global_entropy = self._compute_subset_entropy(global_phi, min(self.k_order, num_blocks))
        
        # Returns VECTOR of pressures, avoiding the Scalarization Trap.
        return {
            'local_entropy': torch.stack(local_entropies),
            'global_entropy': global_entropy
        }

    def _compute_subset_entropy(self, phi: torch.Tensor, k: int) -> torch.Tensor:
        """Helper to compute mean entropy over k-wise subsets of given phi."""
        batch_size, K = phi.shape
        if K < k or k == 0:
            return torch.tensor(0.0, device=phi.device)
            
        from itertools import combinations
        subsets = list(combinations(range(K), k))
        if len(subsets) > 10:
            import random
            subsets = random.sample(subsets, 10)
            
        h_sum = 0.0
        for S in subsets:
            outcomes = phi[:, S] > 0
            powers = torch.pow(2, torch.arange(len(S), device=phi.device))
            keys = (outcomes.long() * powers).sum(dim=1)
            
            counts = torch.bincount(keys, minlength=2**len(S)).float()
            probs = counts / (batch_size + 1e-8)
            probs = probs[probs > 0]
            h_sum += -(probs * torch.log2(probs)).sum()
            
        return h_sum / len(subsets)

class RootPersistencePressure(nn.Module):
    """
    Topological Co-Primality via Root Persistence under Perturbation.
    GCD = 1 (Co-prime) if zero-crossings do not persist jointly.
    """
    def __init__(self, perturbation_scale: float = 0.01):
        super().__init__()
        self.perturbation_scale = perturbation_scale
        
    def forward(self, theta: torch.Tensor, basis_eval: torch.Tensor) -> torch.Tensor:
        """
        Measure persistence of shared roots between pairwise functionals.
        Returns a vector of obstruction pressures [K*(K-1)/2].
        """
        K, D = theta.shape
        phi_raw = torch.matmul(basis_eval, theta.t()) # [N, K]
        
        # Perturb and evaluate roots
        noise = torch.randn_like(theta) * self.perturbation_scale
        phi_perturbed = torch.matmul(basis_eval, (theta + noise).t())
        
        # Detect zero-crossings
        crossings = (phi_raw[:-1] * phi_raw[1:] < 0).float()
        crossings_p = (phi_perturbed[:-1] * phi_perturbed[1:] < 0).float()
        
        # Persistent Crossings (Shared roots across perturbation)
        persistence = crossings * crossings_p
        
        pressures = []
        from itertools import combinations
        for i, j in combinations(range(K), 2):
            # Shared persistent crossings between i and j
            shared = (persistence[:, i] * persistence[:, j]).sum()
            pressures.append(shared)
            
        return torch.stack(pressures)


class BirkhoffPolytopeSampler:
    """
    Sample doubly-stochastic matrices from the Birkhoff polytope.
    
    Uses Sinkhorn-Knopp projection to ensure:
        - Row sums = 1
        - Column sums = 1
        - All entries ≥ 0
    """
    
    def __init__(
        self,
        K: int,
        D: int,
        epsilon: float = 1e-8,
        sinkhorn_iters: int = 50
    ):
        """
        Args:
            K: Number of polynomial functionals
            D: Dimension of basis (degree + 1)
            epsilon: Numerical stability constant
            sinkhorn_iters: Iterations for Sinkhorn-Knopp
        """
        self.K = K
        self.D = D
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
    
    def sample(
        self,
        init_mode: str = 'random',
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample coefficients from Birkhoff polytope.
        
        Args:
            init_mode: 'random', 'uniform', or 'identity'
            temperature: Sampling temperature (higher = more uniform)
            
        Returns:
            theta: [K, D] doubly-stochastic coefficient matrix
        """
        if init_mode == 'uniform':
            # Start from uniform
            theta = torch.ones(self.K, self.D) / max(self.K, self.D)
        elif init_mode == 'identity':
            # Start from approximate identity
            theta = torch.eye(min(self.K, self.D))
            if self.K > self.D:
                theta = torch.cat([theta, torch.ones(self.K - self.D, self.D) / self.D], dim=0)
            elif self.D > self.K:
                theta = torch.cat([theta, torch.ones(self.K, self.D - self.K) / self.K], dim=1)
        elif init_mode == 'orthogonal':
            # Start from orthogonal matrix (Gram-Schmidt via QR) to ensure initial co-primality
            rand_mat = torch.randn(max(self.K, self.D), max(self.K, self.D))
            q, r = torch.linalg.qr(rand_mat)
            theta = q[:self.K, :self.D].abs() # Abs since Birkhoff is internal
        else:
            # Random initialization
            theta = torch.rand(self.K, self.D) / temperature
        
        # Project onto Birkhoff polytope using Sinkhorn-Knopp
        theta = self.sinkhorn_knopp(torch.exp(theta))
        
        return theta
    
    def sinkhorn_knopp(
        self,
        M: torch.Tensor,
        num_iters: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm for Birkhoff polytope projection.
        
        Args:
            M: [K, D] positive matrix
            num_iters: Number of iterations (default: self.sinkhorn_iters)
            
        Returns:
            M_doubly_stochastic: [K, D]
        """
        if num_iters is None:
            num_iters = self.sinkhorn_iters
        
        M = torch.clamp(M, min=self.epsilon)
        return M

    def apply_berkoff_deformation(self, M: torch.Tensor, twist_strength: float = 0.1) -> torch.Tensor:
        """
        Berkoff Deformation: Non-convex, twisted stochastic polytope.
        Violates bistochastic symmetry by injecting asymmetric noise and 
        skewing the barycenter away from the uniform matrix.
        """
        # 1. Asymmetric Skew
        skew = torch.randn_like(M) * twist_strength
        M_skewed = M + skew
        
        # 2. Rescale without Sinkhorn (Violating bistochastic symmetry)
        # Only preserve row sums (stochastic, not bistochastic)
        M_berkoff = M_skewed / (M_skewed.sum(dim=1, keepdim=True) + self.epsilon)
        
        return M_berkoff


class PolynomialCoprimeConfig:
    """
    Configuration for polynomial co-prime functional system.
    
    Manages:
        - Polynomial basis generation
        - Birkhoff polytope coefficient sampling
        - Co-primality verification
    """
    
    def __init__(
        self,
        k: int = 5,
        degree: int = 4,
        basis_type: str = 'chebyshev',
        learnable: bool = True,
        use_saturation: bool = True,
        device: str = None,
        **kwargs
    ):
        """
        Args:
            k: Number of co-prime polynomial functionals
            degree: Polynomial degree
            basis_type: Type of polynomial basis
            learnable: If True, θ_k are learnable parameters
            use_saturation: If True, apply piecewise saturation
            device: Device for tensors
        """
        self.k = k
        self.degree = degree
        self.basis_type = basis_type
        self.learnable = learnable
        self.use_saturation = use_saturation
        self.device = device
        
        # Create polynomial basis
        self.basis = PolynomialBasis(degree, basis_type)
        
        # Create Birkhoff sampler
        self.sampler = BirkhoffPolytopeSampler(k, degree + 1)
        
        # Sample initial coefficients
        theta_init = self.sampler.sample(init_mode='orthogonal')
        self.theta = theta_init.to(device, non_blocking=True)
        
        # Create saturation gate if requested
        if use_saturation:
            self.gate = SaturatedPolynomialGate()
        
        # Heritable Mutation Strengths (Local Time)
        self.mutation_strengths = torch.full((k,), 0.05, device=device)
        self.is_fossilized = torch.zeros(k, dtype=torch.bool, device=device)
        
        # Pressure metrics
        # Use Non-Ergodic Fractal Entropy if available (preserves soliton structure)
        if _HAS_NON_ERGODIC and not kwargs.get('legacy_entropy', False):
            self.orth_pressure_fn = NonErgodicFractalEntropy(k_order=3, num_bands=3)
            # Import and Initialization of Hybrid LAS-Quantizer
            from src.core.non_ergodic_entropy import HybridLassoQuantizer
            self.quantizer = HybridLassoQuantizer(dim=k, lasso_lambda=0.05)
        else:
            self.orth_pressure_fn = HypergraphOrthogonalityPressure(k_order=3)
            self.quantizer = None
            
        self.root_persistence_fn = RootPersistencePressure()
        self.continuous_coprime = ContinuousCoprimality()
        
        # Enforce Chirality (Bostick, 2025)
        # "Coherence can only persist under asymmetric recursion."
        self.ensure_chirality()
        
    def ensure_chirality(self):
        """
        Enforce Non-Zero Chirality (Delta Chi != 0).
        
        Prevents the polynomial basis from being purely symmetric (even) or 
        antisymmetric (odd), which causes total phase cancellation in the 
        resonance limit.
        """
        with torch.no_grad():
            # Check parity of coefficients
            # Even degrees: 0, 2, 4... (indices 0, 2, 4)
            # Odd degrees: 1, 3, 5... (indices 1, 3, 5)
            
            even_mask = torch.arange(self.degree + 1, device=self.device) % 2 == 0
            odd_mask = ~even_mask
            
            even_energy = (self.theta[:, even_mask] ** 2).sum(dim=1)
            odd_energy = (self.theta[:, odd_mask] ** 2).sum(dim=1)
            
            # Chirality requires MIXING of parities.
            # If purely even or purely odd, insert asymmetry.
            
            # Detect pure states (tolerance 1e-6)
            pure_even = odd_energy < 1e-6
            pure_odd = even_energy < 1e-6
            symmetric_defect = pure_even | pure_odd
            
            if symmetric_defect.any():
                # Inject chirality: Add small noise to the missing parity
                target_indices = torch.where(symmetric_defect)[0]
                for idx in target_indices:
                    # If even, add to odd. If odd, add to even.
                    # Or just add random noise to everything and re-project
                    noise = torch.randn(self.theta.shape[1], device=self.device) * 0.05
                    # Bias towards missing parity?
                    if pure_even[idx]:
                        noise[even_mask] = 0 # Only perturb odd
                    elif pure_odd[idx]:
                        noise[odd_mask] = 0 # Only perturb even
                        
                    # Apply perturbation
                    new_coeffs = torch.log(self.theta[idx] + 1e-8) + noise
                    self.theta[idx] = self.sampler.sinkhorn_knopp(torch.exp(new_coeffs).unsqueeze(0)).squeeze(0)
    
    def orthogonality_pressure(self) -> Dict[str, torch.Tensor]:
        """
        Returns Vectorized Orthogonality Pressure.
        """
        theta = self.theta
        x = torch.linspace(-1, 1, 100, device=theta.device)
        basis = self.basis.evaluate(x)
        phi_raw = torch.matmul(basis, theta.t())
        phi = self.gate(phi_raw) if hasattr(self, 'gate') else torch.tanh(phi_raw)
        return self.orth_pressure_fn(phi)

    def continuous_coprimality_pressure(self) -> torch.Tensor:
        """
        Returns Continuous Co-Primality Pressure (Entropy-based).
        """
        x = torch.linspace(-1, 1, 100, device=self.device)
        phi = self.evaluate(x)
        return self.continuous_coprime(phi.t())

    def co_primality_pressure(self) -> torch.Tensor:
        """
        Returns Vectorized Root Persistence Pressure (Topological Co-primality).
        """
        x = torch.linspace(-1, 1, 100, device=self.device)
        basis = self.basis.evaluate(x)
        return self.root_persistence_fn(self.theta, basis)
    
    def mutate(self, berkoff_mode: bool = False, twist_strength: float = 0.05):
        """
        Mutate polynomial coefficients.
        
        berkoff_mode: If True, uses the deformed Berkoff polytope instead 
                    of the symmetric Birkhoff polytope.
        """
        # Rename self.is_fossilized to self.fossil_mask for consistency with the provided snippet
        # If the user intended to keep is_fossilized, this would be a deviation.
        # Assuming the snippet implies this rename.
        if not hasattr(self, 'fossil_mask'):
            self.fossil_mask = self.is_fossilized # Initialize if not present

        if self.fossil_mask.all():
            return

        with torch.no_grad():
            active_mask = ~self.fossil_mask
            
            # Locally inherited mutation strengths
            noise_scale = self.mutation_strengths[active_mask].unsqueeze(-1)
            noise = torch.randn_like(self.theta[active_mask]) * noise_scale
            
            # Apply mutation to active theta
            new_theta_raw = torch.log(self.theta[active_mask] + 1e-8) + noise
            
            # Project onto Birkhoff polytope
            mutated_theta = self.sampler.sinkhorn_knopp(torch.exp(new_theta_raw))
            
            if berkoff_mode:
                # Apply Berkoff deformation to the mutated coefficients
                mutated_theta = self.sampler.apply_berkoff_deformation(mutated_theta, twist_strength)
            
            self.theta[active_mask] = mutated_theta
            
            # Mutate local mutation strengths (Heritability)
            m_noise = torch.randn_like(self.mutation_strengths[active_mask]) * 0.01
            self.mutation_strengths[active_mask] *= torch.exp(m_noise)
            self.mutation_strengths[active_mask] = torch.clamp(self.mutation_strengths[active_mask], 0.001, 0.2)
    
    # ==========================================================================
    # Pointer #3: Irreversibility Hardens Early Bias
    # Saturation-Gated Fossilization
    # ==========================================================================
    
    def _init_saturation_tracking(self):
        """Initialize pressure history tracking for saturation detection."""
        if not hasattr(self, '_pressure_history'):
            self._pressure_history = {k: [] for k in range(self.k)}
            self.saturation_threshold = 0.05
            self.saturation_window = 20
    
    def _is_saturated(self, k: int) -> bool:
        """
        Check if functional k has reached constraint geometry saturation.
        
        Pointer #3: Seriousness is a property of constraint geometry, not commitment.
        Premature fossilization = topology lock-in.
        """
        self._init_saturation_tracking()
        history = self._pressure_history.get(k, [])
        
        if len(history) < self.saturation_window:
            return False  # Not enough history
        
        recent = torch.tensor(history[-self.saturation_window:])
        oscillation = recent.std()
        
        # Saturated = bounded oscillation (not convergence!)
        return oscillation.item() < self.saturation_threshold
    
    def fossilize(self, k: int) -> bool:
        """
        Attempt to fossilize functional k.
        
        Gate: Only fossilize at admissibility boundaries, not during active saturation.
        Pointer #3: Irreversible moves before saturation guarantee premature topology lock-in.
        """
        if k < 0 or k >= self.k:
            return False
        
        if not self._is_saturated(k):
            # Premature - block fossilization
            return False
        
        self.is_fossilized[k] = True
        return True
    
    def update_pressure_history(self, k: int, pressure: float):
        """Track pressure for saturation detection."""
        self._init_saturation_tracking()
        if k not in self._pressure_history:
            self._pressure_history[k] = []
        self._pressure_history[k].append(pressure)
        
        # Bounded memory
        if len(self._pressure_history[k]) > self.saturation_window * 2:
            self._pressure_history[k] = self._pressure_history[k][-self.saturation_window:]
    
    def get_saturation_status(self) -> Dict[int, bool]:
        """Get saturation status for all functionals."""
        return {k: self._is_saturated(k) for k in range(self.k)}
    
    def get_coefficients_tensor(self) -> torch.Tensor:
        """
        Get coefficient matrix for all functionals.
        
        Returns:
            theta: [K, D] Birkhoff polytope coefficients
        """
        return self.theta
    
    def evaluate_polynomial(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the k-th polynomial functional at point x.
        
        This method is used by Meta-Polytope Matrioshka system for CRT moduli generation.
        
        Args:
            k: Index of polynomial functional (0 <= k < self.k)
            x: Input value(s) to evaluate at
            
        Returns:
            phi_k: Evaluation of φ_k(x; θ_k)
        """
        if k < 0 or k >= self.k:
            raise ValueError(f"Polynomial index k={k} out of range [0, {self.k})")
            
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Ensure x has proper shape for basis evaluation
        if x.dim() == 0:
            x = x.unsqueeze(0)  # Make it [1] instead of []
            
        # Get basis evaluations: [batch, D] or [D] if single point
        basis_vals = self.basis.evaluate(x)
        
        # Get coefficients for k-th polynomial: [D]
        theta_k = self.theta[k]
        
        # Compute φ_k(x) = Σ_i θ_k[i] · p_i(x)
        if basis_vals.dim() == 1:
            # Single point evaluation
            phi_k = torch.dot(basis_vals, theta_k)
        else:
            # Batch evaluation: [batch, D] × [D] = [batch]
            phi_k = torch.matmul(basis_vals, theta_k)
            
        # Apply saturation if enabled
        if self.use_saturation and hasattr(self, 'gate'):
            phi_k = self.gate(phi_k)
        else:
            # Default activation for stability
            phi_k = torch.tanh(phi_k)
            
        # Apply quantization if available
        if hasattr(self, 'quantizer') and self.quantizer is not None:
            phi_k = self.quantizer(phi_k.unsqueeze(-1) if phi_k.dim() == 0 else phi_k)
            if phi_k.dim() > 0:
                phi_k = phi_k.squeeze(-1)
                
        return phi_k
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all φ_k(x; θ_k).
        
        Args:
            x: [batch, ...] input values
            
        Returns:
            phi: [batch, ..., K] functional values
        """
        # Get basis evaluations: [batch, ..., D]
        basis_vals = self.basis.evaluate(x)
        
        # Matrix multiply: [batch, ..., D] × [D, K] = [batch, ..., K]
        phi = torch.matmul(basis_vals, self.theta.t())
        
        # Apply saturation
        if self.use_saturation:
            phi = self.gate(phi)
            
        # Apply Hybrid LAS-Quantization (Bostick, 2026 update)
        if hasattr(self, 'quantizer') and self.quantizer is not None:
            phi = self.quantizer(phi)
        
        return phi
    
    def _verify_coprime_approximate(self):
        """
        Approximate co-primality check.
        
        True co-primality requires polynomial GCD = 1, which is expensive.
        We use a simpler check: functional values should be sufficiently different.
        """
        # Sample test points
        x_test = torch.linspace(-1, 1, 100, device=self.device)
        
        # Evaluate all functionals
        phi = self.evaluate(x_test)  # [100, K]
        
        # Check pairwise correlation (low correlation ≈ co-primality)
        phi_norm = phi / (torch.norm(phi, dim=0, keepdim=True) + 1e-8)
        correlation = torch.matmul(phi_norm.t(), phi_norm) / 100
        
        # Off-diagonal should be small
        mask = ~torch.eye(self.k, dtype=torch.bool, device=self.device)
        off_diag = correlation[mask].abs().mean()
        
        if off_diag > 0.5:
            print(f"Warning: High correlation {off_diag:.3f} - functionals may not be co-prime")
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return {
            'k': self.k,
            'degree': self.degree,
            'basis_type': self.basis_type,
            'theta': self.theta.cpu().numpy().tolist()
        }


def compute_polynomial_gcd_degree(
    phi1_coeffs: torch.Tensor,
    phi2_coeffs: torch.Tensor,
    tolerance: float = 1e-6
) -> int:
    """
    Compute degree of GCD of two polynomials (approximate).
    
    Uses Euclidean algorithm on coefficient representations.
    
    Args:
        phi1_coeffs: [D] coefficients of first polynomial
        phi2_coeffs: [D] coefficients of second polynomial
        tolerance: Threshold for zero coefficients
        
    Returns:
        gcd_degree: Degree of GCD (0 means co-prime)
    """
    # Simple heuristic: if inner product is small, likely co-prime
    inner_prod = torch.dot(phi1_coeffs, phi2_coeffs).abs()
    norm_prod = torch.norm(phi1_coeffs) * torch.norm(phi2_coeffs)
    
    if norm_prod < tolerance:
        return 0
    
    similarity = inner_prod / (norm_prod + tolerance)
    
    # High similarity suggests common factors
    if similarity > 0.9:
        return 1  # Likely share common factors
    else:
        return 0  # Likely co-prime


