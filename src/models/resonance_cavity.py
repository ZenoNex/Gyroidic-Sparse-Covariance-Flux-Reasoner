"""
Resonance cavity memory module for persistent self-model storage.

Modular resonant cavity per prime field with introspection integration
and GDPO-enhanced residue pattern storage.

Author: William Matthew Bryant
Enhanced: January 2026 with GDPO integration
"""

import torch
import torch.nn as nn
import math
import hashlib
import struct
from typing import List, Dict, Optional, Tuple


class GyroidicFluxAlignment(nn.Module):
    """
    Speculative Primitive: Gyroidic Flux Alignment Operator.
    
    "Bends" residue weights toward low-violation paths based on local manifold curvature.
    w_hat = w * exp( - Violation / Flux_Integral )
    """
    def __init__(self, dim: int, diffusivity_init: float = 0.5):
        super().__init__()
        self.dim = dim
        self.diffusivity = nn.Parameter(torch.tensor(diffusivity_init)) # kappa
        
    def forward(self, residue_weights: torch.Tensor, gcve_pressure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residue_weights: [batch, K, D] (or similar) weights
            gcve_pressure: [batch] Gyroid violation
        """
        if gcve_pressure is None:
            return residue_weights
            
        # Reshape violation for broadcasting: [batch, 1, 1]
        v = gcve_pressure.view(-1, 1, 1) if gcve_pressure.dim() == 1 else gcve_pressure
        
        # Approximate Flux Integral over Manifold M
        # flux ~ kappa * grad(phi)
        # We approximate this denominator as a learned scaling of the weights' energy
        grad_phi_proxy = torch.norm(residue_weights, dim=-1, keepdim=True) # [batch, K, 1]
        flux_integral = self.diffusivity * grad_phi_proxy.mean(dim=1, keepdim=True) + 1e-6
        
        # Warp Factor
        # exp( - V / Flux )
        warp = torch.exp( - v / flux_integral )
        
        return residue_weights * warp


class HeritableTrustVault(nn.Module):
    """
    Symbolic trust: topological cache of successful symbolic partitions.
    Uses residue pattern hashing (no gradients required).
    
    Allows contradictory trusted patterns to coexist until selection.
    """
    def __init__(self, table_size: int = 1024, k_dim: int = 5):
        super().__init__()
        self.table_size = table_size
        self.k_dim = k_dim
        # Store high-survivorship residue vectors: [size, K]
        self.register_buffer('trust_table', torch.zeros(table_size, k_dim))
        self.register_buffer('trust_scores', torch.zeros(table_size))
        self.register_buffer('longevity', torch.zeros(table_size, dtype=torch.long))
        # Persistent salt: stored as buffer so it survives save/load cycles.
        # This guarantees the same residue hashes to the same slot across runs.
        import os
        _salt_bytes = os.urandom(16)
        self.register_buffer(
            '_hash_salt',
            torch.frombuffer(bytes(_salt_bytes), dtype=torch.uint8).clone()
        )

    def _hash(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Hash symbolic residues [batch, K] to [batch] table indices.

        Uses hashlib.sha256 with a persistent per-instance salt so that:
        - The same residue pattern always maps to the same table slot (deterministic).
        - Cross-run determinism is guaranteed because the salt is stored as a buffer
          and restored via state_dict (unlike Python's built-in hash which is
          randomised per process by PYTHONHASHSEED).
        """
        # Detach and move to CPU for byte-level hashing (no gradient flow here)
        r_cpu = residues.detach().cpu()
        salt_bytes = bytes(self._hash_salt.cpu().numpy().tobytes())
        indices = []
        for row in r_cpu:
            # Quantise to int16 to stabilise floating-point jitter before hashing
            row_ints = (row.float().clamp(-3e4, 3e4) * 100).short().numpy().tobytes()
            digest = hashlib.sha256(salt_bytes + row_ints).digest()
            # Take first 8 bytes as a uint64 and reduce modulo table_size
            slot = struct.unpack_from('<Q', digest)[0] % self.table_size
            indices.append(slot)
        return torch.tensor(indices, dtype=torch.long, device=residues.device)

    def update(self, residues: torch.Tensor, survivorship: torch.Tensor):
        """
        Update trust table with surviving patterns.
        Contradictory patterns are allowed via soft-overwriting or separate slots.
        """
        indices = self._hash(residues)
        for i, idx in enumerate(indices):
            # Trust is heritable: surviving patterns amplify their slot's survival
            # We allow slight variation by blending residues if trust is high,
            # but preserve contradictions by only updating if trust is significantly higher.
            if survivorship[i] > self.trust_scores[idx]:
                if residues.shape[1] == self.k_dim:
                    self.trust_table[idx] = residues[i]
                else:
                    self.trust_table[idx] = residues[i].mean()
                
                self.trust_scores[idx] = survivorship[i]
            
            self.longevity[idx] += 1

    def query(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Retrieve trust scores for queried residue patterns.
        """
        indices = self._hash(residues)
        return self.trust_scores[indices]


class BreatherMode(nn.Module):
    """
    Localized Breather Mode (RIC Eq 6).
    
    Implements sine-Gordon breather solitons as concept packet storage:
        u_n(x,t) = 4*arctan[ (sqrt(1-w^2)/w) * (1/cosh(sqrt(1-w^2)*(x-x_n))) * sin(w*t) ]
    
    Properties:
        - Localized: doesn't spread (concept coherence)
        - Periodic: doesn't decay (persistent memory)
        - Survives collisions intact (concept packets don't destroy each other)
    """
    def __init__(self, num_breathers: int = 10, dim: int = 64):
        super().__init__()
        self.num_breathers = num_breathers
        self.dim = dim
        
        # Breather frequencies: omega_n = 1/p_n (inverse primes ensure omega < 1)
        primes = self._generate_primes(num_breathers)
        omegas = 1.0 / primes.float()
        self.register_buffer('omegas', omegas)  # [num_breathers]
        
        # Center positions: evenly spaced across the manifold
        centers = torch.linspace(-math.pi, math.pi, num_breathers)
        self.register_buffer('centers', centers)  # [num_breathers]
        
        # Precompute sqrt(1 - omega^2) for stability
        sqrt_term = torch.sqrt(1.0 - omegas ** 2)
        self.register_buffer('sqrt_term', sqrt_term)  # [num_breathers]
        
        # Amplitude projection: maps breather output to hidden_dim
        self.projection = nn.Linear(num_breathers, dim, bias=False)
        
        # Internal clock
        self.register_buffer('t', torch.tensor(0.0))
    
    def _generate_primes(self, n: int) -> torch.Tensor:
        """Generates the first n primes."""
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
        return torch.tensor(primes, dtype=torch.long)
    
    def evaluate(self, x: torch.Tensor, t: float = None) -> torch.Tensor:
        """
        Evaluate all breather modes at positions x and time t.
        
        Args:
            x: [batch, spatial_dim] or [spatial_dim] positions along the manifold
            t: Time (uses internal clock if None)
        
        Returns:
            breather_field: [batch, num_breathers] breather amplitudes
        """
        if t is None:
            t = self.t.item()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, spatial_dim]
        
        # Use mean of spatial coordinates as 1D position proxy
        x_pos = x.mean(dim=-1)  # [batch]
        
        # u_n(x,t) = 4*arctan[ (sqrt(1-w^2)/w) * sech(sqrt(1-w^2)*(x-x_n)) * sin(w*t) ]
        results = []
        for n in range(self.num_breathers):
            w = self.omegas[n]
            sq = self.sqrt_term[n]
            x_n = self.centers[n]
            
            # Amplitude ratio
            amp_ratio = sq / (w + 1e-8)
            
            # Sech envelope (localization)
            sech_arg = sq * (x_pos - x_n)
            sech_val = 1.0 / (torch.cosh(sech_arg) + 1e-8)
            
            # Temporal oscillation
            sin_val = torch.sin(w * t)
            
            # Full breather
            u_n = 4.0 * torch.arctan(amp_ratio * sech_val * sin_val)
            results.append(u_n)
        
        return torch.stack(results, dim=-1)  # [batch, num_breathers]
    
    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Advance breather clock and return projected breather field.
        
        Args:
            x: [batch, dim] state positions
            dt: Time step
        
        Returns:
            breather_contribution: [batch, dim] projected breather excitations
        """
        self.t += dt
        amplitudes = self.evaluate(x)  # [batch, num_breathers]
        return self.projection(amplitudes)  # [batch, dim]

class ResonanceCavity(nn.Module):
    """
    Extra-cavity resonator for Heritable Trust and residue pattern memory.
    
    Update rule (Signal Sovereignty-enhanced):
        dM/dt = -γM + Σ A_ij^H · R_ij + κ · introspection_direction + η · residue_patterns
        
    Stores:
        - Validated introspective directions (moral, creative, metacognitive)
        - Stable residue patterns from SignalSovereignty-reconstructed samples
        - Per-prime field importance weights (Heritable Trust)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_modes: int = 64,
        poly_config: Optional['PolynomialCoprimeConfig'] = None,
        decay_rate: float = 0.1,
        introspection_weight: float = 0.5,
        residue_weight: float = 0.05,
        violation_weight: float = 0.4, # Weight for GCVE violation feedback
        modular: bool = True,
        track_residues: bool = True
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_modes: Number of resonant modes
            poly_config: Polynomial co-prime configuration (replaces primes)
            decay_rate: γ, decay rate for memory
            introspection_weight: κ, weight for introspection feedback
            residue_weight: η, weight for residue pattern feedback
            violation_weight: λ_v, weight for topological violation feedback
            modular: If True, maintain separate cavity per prime field
            track_residues: If True, store residue patterns for the KL prior
        """
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        # Extract K from polynomial config or use default
        if poly_config is not None:
            self.poly_config = poly_config
            self.K = poly_config.k if modular else 1
        else:
            # Fallback: create default config
            from src.core.polynomial_coprime import PolynomialCoprimeConfig
            self.poly_config = PolynomialCoprimeConfig(k=5, degree=4)
            self.K = 5 if modular else 1
        self.decay_rate = decay_rate
        self.introspection_weight = introspection_weight
        self.residue_weight = residue_weight
        self.violation_weight = violation_weight
        self.modular = modular
        self.track_residues = track_residues
        
        # 1. Memory state: [K, num_modes, hidden_dim]
        # Each prime field has its own resonant modes
        self.register_buffer(
            'M',
            torch.randn(self.K, num_modes, hidden_dim) * 0.01
        )
        
        # 2. Dark Matter Store (Speculative Latent states)
        # Counterfactual/unrealized topological flows
        self.register_buffer(
            'D_dark',
            torch.zeros(self.K, num_modes, hidden_dim)
        )
        
        # Symbolic pattern hashing
        if track_residues:
            self.trust_vault = HeritableTrustVault(
                table_size=num_modes * self.K,
                k_dim=self.poly_config.k
            )
        
        # Mutation bias based on trust (heritability)
        self.register_buffer('mutation_bias', torch.ones(self.K))
        
        # Learnable coupling matrices
        self.coupling = nn.ModuleList([
            nn.Linear(hidden_dim, num_modes, bias=False)
            for _ in range(self.K)
        ])
        
        # Residue stores for prior calculation
        self.register_buffer(
            'residue_memory',
            torch.zeros(self.K, num_modes, self.poly_config.degree + 1)
        )
        self.register_buffer(
            'pattern_confidence',
            torch.ones(self.K, num_modes)
        )
        
        # 5. Breather Modes (RIC Eq 6): Localized concept packets
        self.breather = BreatherMode(
            num_breathers=min(10, self.K * 2),
            dim=hidden_dim
        )
        # 6. Flux Alignment
        self.flux_alignment = GyroidicFluxAlignment(dim=hidden_dim)
        
    @torch.no_grad()
    def update(
        self,
        attention_states: torch.Tensor,
        introspection_directions: Optional[torch.Tensor] = None,
        expected_residues: Optional[torch.Tensor] = None,
        gcve_pressures: Optional[torch.Tensor] = None,
        field_idx: int = 0,
        dt: float = 0.1,
        reconstruction_pressure: Optional[torch.Tensor] = None,
        refined_residues: Optional[torch.Tensor] = None,
        instability_severity: float = 0.0
    ):
        """
        Update resonance cavity memory (GDPO-enhanced + GCVE + System 2 Feedback).
        
        Args:
            attention_states: [batch, seq_len, hidden_dim] attention outputs
            introspection_directions: Optional [batch, hidden_dim] validated directions
            expected_residues: Optional [batch, K] System 1 guess
            gcve_pressures: Optional [batch, num_windows] GCVE violation scores
            field_idx: Which prime field (0 to K-1)
            dt: Time step for update
            reconstruction_pressure: Optional [batch] CRT reconstruction errors
            refined_residues: Optional [batch, K] System 2 (Physics) ground truth
            instability_severity: Normalized [0, 1] indicator of topological panic (aborts)
        """
        if field_idx >= self.K:
            field_idx = 0
        
        batch_size, seq_len, _ = attention_states.shape
        
        # Decide which residues to store: Physics (Refined) > Intuition (Expected)
        # "System 2 trains System 1's memory"
        target_residues_for_storage = refined_residues if refined_residues is not None else expected_residues
        
        # Decay term: -γM
        decay_term = -self.decay_rate * self.M[field_idx]  # [num_modes, hidden_dim]
        
        # Attention-driven excitation: Σ A_ij^H · R_ij
        # Compute coupling between attention states and modes
        coupling_weights = self.coupling[field_idx](
            attention_states.mean(dim=1)  # [batch, hidden_dim]
        )  # [batch, num_modes]
        
        # Weighted sum of attention states
        excitation = torch.einsum(
            'bm,bh->mh',
            coupling_weights,
            attention_states.mean(dim=1)
        )  # [num_modes, hidden_dim]
        excitation = excitation / (batch_size + 1e-8)
        
        # Introspection feedback: κ · introspection_direction
        introspection_term = torch.zeros_like(self.M[field_idx])
        if introspection_directions is not None:
            # Broadcast introspection direction to all modes
            introspection_avg = introspection_directions.mean(dim=0)  # [hidden_dim]
            introspection_term = self.introspection_weight * introspection_avg.unsqueeze(0)
            # [1, hidden_dim] -> broadcast to [num_modes, hidden_dim]
            introspection_term = introspection_term.expand(self.num_modes, -1)
            
        # GCVE Violation feedback: λ_v · gcve_pressures
        # High violation -> amplify "defect modes" to force attention to resolve topology
        violation_term = torch.zeros_like(self.M[field_idx])
        if gcve_pressures is not None:
             # Aggregate violation severity [batch]
             mean_violation = gcve_pressures.mean() 
             
             # INSTABILITY AWARENESS:
             # If system is actively aborting (instability_severity > 0),
             # we Amplify the violation signal to force a state shift.
             if instability_severity > 0:
                 mean_violation *= (1.0 + instability_severity * 2.0)
                 
             # Add isotropic excitation proportional to violation
             # (In a full version, this would be mode-specific, but isotropic is a good start)
             violation_term = self.violation_weight * mean_violation * torch.ones_like(self.M[field_idx])
        
        # Residue pattern feedback: η · residue_patterns (Symbolic Hashing)
        # INSTABILITY GATING: Do not memorize patterns if system is in topological panic!
        if self.track_residues and target_residues_for_storage is not None and instability_severity < 0.5:
            # Store high-quality residue patterns (low reconstruction pressure)
            if reconstruction_pressure is not None:
                # Survivorship index based on inverse pressure
                survivorship = 1.0 / (reconstruction_pressure + 1e-4)
                
                # Update trust vault with symbolic residue vectors
                self.trust_vault.update(
                    target_residues_for_storage,
                    survivorship
                )
        
        # Harmonic-Differential Equivalence Map (Bostick, 2025)
        # dC/dt = Gamma * C^n - lambda * C + eta * (Grad_S . Grad_Omega)
        # Mapping:
        # C      -> self.M[field_idx] (Memory State / Coherence Density)
        # Gamma  -> excitation (Positive Feedback from Attention)
        # lambda -> self.decay_rate (Leakage)
        # eta    -> self.violation_weight (Coupling Gain)
        # Grad_S -> mean_violation (Entropy Gradient)
        # Grad_O -> introspection_term (Possibility Gradient)
        
        Gamma_term = excitation
        lambda_term = -self.decay_rate * self.M[field_idx]
        eta_term = violation_term + introspection_term # Combined coupling fields
        
        # --- MISCHIEF BAND (Documentation Inspired) ---
        # Reward topological violations with "mischief" to prevent collapse.
        # When mean_violation is low (Play), we inject more mischief to explore.
        mischief_strength = 0.05 * (1.0 - torch.tanh(mean_violation if 'mean_violation' in locals() else torch.tensor(0.0)))
        mischief_noise = torch.randn_like(self.M[field_idx]) * mischief_strength
        
        # dC/dt = Gamma + lambda + eta + mischief
        dC_dt = Gamma_term + lambda_term + eta_term + mischief_noise
        
        # --- BREATHER MODE INTEGRATION (RIC Eq 6) ---
        # Localized breather excitations preserve concept packets as
        # standing-wave interference patterns within the cavity.
        breather_input = self.M[field_idx].mean(dim=0, keepdim=True)  # [1, hidden_dim]
        breather_excitation = self.breather(breather_input, dt=dt)  # [1, hidden_dim]
        # Scale breather contribution to avoid overwhelming the decay dynamics
        dC_dt = dC_dt + 0.05 * breather_excitation.expand_as(dC_dt)
        
        # Update Memory for Non-Teleological Flow
        # dt represents the "Manifold Clock"
        self.M[field_idx] = self.M[field_idx] + dt * dC_dt
        
        # --- Speculative Neuroscience Layer: Dark Matter Accumulation ---
        # Dark matter captures the "unreconciled" residues (the holes).
        # It evolves slower and represents counterfactual potential.
        if refined_residues is not None and expected_residues is not None:
             residue_gap = torch.abs(refined_residues - expected_residues).mean()
             if residue_gap > 0.1:
                  # Inject gap into dark matter as a "speculative trace"
                  dD_dt = (excitation - self.M[field_idx]) * residue_gap
                  self.D_dark[field_idx] += dt * 0.1 * dD_dt
        
        # Normalize to prevent blow-up (Lipschitz Bound enforcement)
        # But allow small violations to preserve resonance vibrancy
        # (Soft normalization instead of hard clipping/division if possible)
        norm = torch.norm(self.M[field_idx], dim=-1, keepdim=True)
        self.M[field_idx] = self.M[field_idx] / (norm + 1e-8)
        
        # Dark matter damping (gradual fossilization)
        self.D_dark[field_idx] *= 0.99
    
    def prime_harmonic_field(
        self,
        t: float,
        field_idx: int = 0,
        num_harmonics: int = 20
    ) -> torch.Tensor:
        """
        Prime-Anchored Harmonic Field (RIC Eq 5).
        
        F(t) = Σ_{n=1}^{N} a_n · sin(2π · p_n · t + φ_n)
        
        where:
            a_n: Amplitude derived from cavity mode energy (evolved, not learned)
            p_n: n-th prime number (via PrimeResonanceLadder)
            φ_n: Phase offset (derived from memory state, accumulates like Berry phase)
        
        This field is the "carrier wave" of the system's cognitive state.
        Peaks at certain prime harmonics correspond to activated concept clusters.
        
        Args:
            t: Time parameter
            field_idx: Which prime field cavity to derive amplitudes from
            num_harmonics: Number of prime harmonics to superpose
            
        Returns:
            field_value: Scalar field evaluation F(t)
        """
        from src.core.fgrt_primitives import PrimeResonanceLadder
        
        if field_idx >= self.K:
            field_idx = 0
        
        # Get prime frequencies: f_{p_n} = 2π·ln(p_n)
        ladder = PrimeResonanceLadder(num_resonators=num_harmonics)
        freqs = ladder.forward().to(self.M.device)  # [num_harmonics]
        
        # Derive amplitudes from cavity mode energies (evolved, not learned)
        # a_n = ||M[field_idx, n, :]|| for modes that exist, else decay
        M_field = self.M[field_idx]  # [num_modes, hidden_dim]
        mode_energies = torch.norm(M_field, dim=-1)  # [num_modes]
        
        # Map num_modes → num_harmonics via adaptive pooling or truncation
        if mode_energies.shape[0] >= num_harmonics:
            amplitudes = mode_energies[:num_harmonics]
        else:
            # Pad with decaying amplitudes
            padding = torch.zeros(
                num_harmonics - mode_energies.shape[0],
                device=self.M.device
            )
            amplitudes = torch.cat([mode_energies, padding])
        
        # Derive phase offsets from memory state structure
        # φ_n = angle of mean of n-th mode vector (Berry-like accumulated phase)
        phases = torch.zeros(num_harmonics, device=self.M.device)
        for n in range(min(num_harmonics, self.num_modes)):
            mode_vec = M_field[n]  # [hidden_dim]
            # Interpret first two components as complex pair for phase extraction
            if mode_vec.shape[0] >= 2:
                phases[n] = torch.atan2(mode_vec[1], mode_vec[0])
        
        # Evaluate harmonic field: F(t) = Σ a_n · sin(f_n · t + φ_n)
        field_value = torch.sum(amplitudes * torch.sin(freqs * t + phases))
        
        return field_value
    
    def query(
        self,
        query_vector: torch.Tensor,
        field_idx: int = 0,
        top_k: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Query resonance cavity for relevant modes.
        
        Args:
            query_vector: [batch, hidden_dim] query
            field_idx: Which prime field to query
            top_k: Number of top resonant modes to return
            
        Returns:
            Dictionary with:
                - 'modes': [batch, top_k, hidden_dim]
                - 'resonance_scores': [batch, top_k]
        """
        if field_idx >= self.K:
            field_idx = 0
        
        batch_size = query_vector.shape[0]
        
        # Normalize query
        query_norm = query_vector / (torch.norm(query_vector, dim=-1, keepdim=True) + 1e-8)
        
        # Compute resonance with all modes
        modes = self.M[field_idx]  # [num_modes, hidden_dim]
        
        # Cosine similarity
        resonance = torch.mm(query_norm, modes.t())  # [batch, num_modes]
        
        # Get top-k
        top_scores, top_indices = torch.topk(resonance, k=min(top_k, self.num_modes), dim=-1)
        
        # Gather top modes
        top_modes = modes[top_indices]  # [batch, top_k, hidden_dim]
        
        return {
            'modes': top_modes,
            'resonance_scores': top_scores
        }
    
    def get_mutation_bias(self, current_residues: torch.Tensor) -> torch.Tensor:
        """
        Compute mutation bias factors per sample based on trust in the current pattern.
        """
        if not self.track_residues:
            return torch.ones(current_residues.shape[0], device=current_residues.device)
            
        trust = self.trust_vault.query(current_residues)
        # Higher trust -> lower mutation strength (stabilization)
        bias = torch.exp(-trust)
        return bias
    
    def get_residue_prior(
        self,
        field_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get stored residue patterns as prior for GDPO KL regularization.
        
        Args:
            field_idx: Optional specific field, or None for all fields
            
        Returns:
            residue_prior: [K] or [1] expected residue values
            confidence: [K] or [1] confidence scores
        """
        if not self.track_residues:
            # Return default uniform prior
            if field_idx is not None:
                return torch.zeros(1, device=self.M.device), torch.zeros(1, device=self.M.device)
            return torch.zeros(self.K, device=self.M.device), torch.zeros(self.K, device=self.M.device)
        
        if field_idx is not None:
            # Single field - return weighted average over modes
            weights = torch.softmax(self.pattern_confidence[field_idx], dim=0)
            prior = torch.sum(
                weights.unsqueeze(-1) * self.residue_memory[field_idx],
                dim=0
            )
            conf = self.pattern_confidence[field_idx].mean()
            return prior.squeeze(), conf
        else:
            # All fields
            priors = []
            confs = []
            for k in range(self.K):
                weights = torch.softmax(self.pattern_confidence[k], dim=0)
                prior = torch.sum(
                    weights.unsqueeze(-1) * self.residue_memory[k],
                    dim=0
                )
                priors.append(prior.squeeze())
                confs.append(self.pattern_confidence[k].mean())
            
            return torch.stack(priors), torch.stack(confs)
    
    def forward(
        self,
        attention_states: torch.Tensor,
        introspection_directions: Optional[torch.Tensor] = None,
        expected_residues: Optional[torch.Tensor] = None,
        gcve_pressures: Optional[torch.Tensor] = None,
        reconstruction_pressure: Optional[torch.Tensor] = None,
        refined_residues: Optional[torch.Tensor] = None,
        instability_severity: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Update all field cavities and return aggregated memory state.
        
        Args:
            attention_states: [batch, seq_len, hidden_dim]
            introspection_directions: Optional [batch, hidden_dim]
            expected_residues: Optional [batch, K] for residue pattern storage
            gcve_pressures: Optional [batch, num_windows] GCVE violation scores
            reconstruction_pressure: Optional [batch] for quality filtering
            refined_residues: Optional [batch, K] System 2 feedback
            instability_severity: [0, 1] topological panic indicator
            
        Returns:
            Dictionary with:
                - 'memory_state': [batch, K * num_modes, hidden_dim]
                - 'residue_prior': [K] or None
                - 'prior_confidence': [K] or None
        """
        # Update each field
        for k in range(self.K):
            self.update(
                attention_states,
                introspection_directions,
                expected_residues,
                gcve_pressures,
                field_idx=k,
                reconstruction_pressure=reconstruction_pressure,
                refined_residues=refined_residues,
                instability_severity=instability_severity
            )
        
        # Return current memory state (flattened across fields)
        batch_size = attention_states.shape[0]
        memory = self.M.unsqueeze(0).expand(batch_size, -1, -1, -1)
        dark_matter = self.D_dark.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Combined Speculative Output (M + D_dark)
        # Phi_spec = sigma( L + C_res + D_dark )
        speculative_state = memory + 0.1 * dark_matter
        
        memory_flat = speculative_state.reshape(batch_size, self.K * self.num_modes, self.hidden_dim)
        
        # Apply Flux Alignment (Gyroidic Warping)
        if gcve_pressures is not None:
             mean_violation = gcve_pressures.mean(dim=1) if gcve_pressures.dim() > 1 else gcve_pressures
             memory_flat = self.flux_alignment(memory_flat, gcve_pressure=mean_violation)
        
        # Get residue priors for KL regularization
        residue_prior, prior_confidence = self.get_residue_prior()
        
        return {
            'memory_state': memory_flat,
            'dark_matter': dark_matter.reshape(batch_size, -1, self.hidden_dim),
            'residue_prior': residue_prior if self.track_residues else None,
            'prior_confidence': prior_confidence if self.track_residues else None
        }
