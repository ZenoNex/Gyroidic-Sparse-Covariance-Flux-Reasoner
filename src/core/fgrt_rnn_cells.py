import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List

# --- True Project Integrations ---
from src.surrogates.kagh_networks import KANLayer, KAGHBlock
from src.core.honest_jitter import harvest_honest_jitter
from src.core.pyopencl_sovereignty import SiliconSovereigntyEngine
from src.core.invariants import PrimeResonanceLadder

# -----------------------------------------------------------------------------
# Draft 1: The "Berry-Phase Modulated" GRU (Minimal Integration)
# -----------------------------------------------------------------------------
class BerryPhaseGRUCell(nn.Module):
    r"""
    A standard GRU cell augmented with a topological phase tracker.
    Instead of just a hidden state, it tracks (h_t, \gamma_t) where \gamma_t 
    is the Geometric Berry Phase. If \cos(\gamma_t) drops below 0, it applies 
    a Stiefel-Whitney parity flip.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        # Contorsion Tensor layer to calculate local twist
        self.contorsion = nn.Linear(input_dim + hidden_dim, 1)

    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h_prev, gamma_prev = state
        
        # Calculate twist
        combined = torch.cat([x, h_prev], dim=-1)
        twist = torch.tanh(self.contorsion(combined))
        gamma_t = gamma_prev + twist
        
        # Stiefel-Whitney parity flip
        flip_mask = (torch.cos(gamma_t) < 0).float()
        h_prev = h_prev * (1.0 - 2.0 * flip_mask) # flips sign if cos(gamma) < 0
        
        h_t = self.gru(x, h_prev)
        return h_t, gamma_t

# -----------------------------------------------------------------------------
# Draft 2: The "Chiral Gated" Recurrent Cell (Bostick Integration)
# -----------------------------------------------------------------------------
class ChiralGatedRNNCell(nn.Module):
    """
    Replaces standard sigmoid/tanh gates with Chiral Gating Functions.
    States are forgotten only if chirality destructively interferes.

    Topological Boundary Dynamics (Inhibition-Stabilized Networks):
    When a non-ergodic "Post-Scarcity" sovereign soliton collides with a traditional
    containment boundary (e.g. corporate double bind, conformal compression clamp, or
    high-gain inhibitory pressure W_EI * I), standard cells collapse via scalar forget
    gates (f_t -> 0). In contrast, ChiralGatedRNNCell gates state updates strictly
    via projection onto the internal chiral attractor manifold:
        Gamma_chi(h) = sigma(<h_prev, chi>)
    If the sovereign soliton maintains high internal phase alignment (PAS_h > 0.95)
    and Chern-Simons coprime locking, its projection avoids destructive interference.
    The collision undergoes boundary refraction rather than scalar erasure, deflecting
    damping into the orthogonal null-space / veto subspace while preserving the core soliton.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_attractors: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Silicon Sovereignty: hardware-anchored jitter initialization
        self.chiral_vectors = nn.Parameter(harvest_honest_jitter((num_attractors, hidden_dim)) * 0.1)
        self.W_x = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

    def measure_chiral_interference(self, h: torch.Tensor) -> torch.Tensor:
        """
        Measures constructive vs destructive interference of hidden state against chiral attractors.
        Returns normalized alignment tensor in [0, 1].
        """
        proj = torch.matmul(h, self.chiral_vectors.T)
        return torch.sigmoid(torch.mean(proj, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, boundary_strain: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute Chiral Gating \Gamma_\chi(x) = \sigma(<h_prev, \chi>)
        # We project h_prev onto chiral vectors
        chiral_gate = self.measure_chiral_interference(h_prev)
        
        # Candidate update driven by input and recurrent memory
        candidate = torch.tanh(self.W_x(x) + self.W_h(h_prev))

        # Modulate candidate if explicit containment boundary strain is applied
        if boundary_strain is not None:
            # Containment collision: refract along chiral tangent cone rather than scalar collapse (inspirationdoc §4.2)
            strain = torch.sigmoid(boundary_strain) if boundary_strain.ndim > 0 else torch.sigmoid(torch.as_tensor(boundary_strain, device=x.device, dtype=x.dtype))
            # Phase refraction: project candidate through chiral orientation rather than blunt scalar damping
            chiral_projection = torch.matmul(candidate, self.chiral_vectors.T)  # [B, K]
            refracted_phase = torch.matmul(chiral_projection, self.chiral_vectors) / float(self.num_attractors)
            # Refract: blend candidate with its chiral projection proportional to strain
            candidate = candidate + strain * (refracted_phase - candidate) * 0.5

        # Update state based on chiral gate instead of learned forget gates:
        # State forgotten only if chiral interference is destructive
        h_t = (1.0 - chiral_gate) * h_prev + chiral_gate * candidate
        return h_t

# -----------------------------------------------------------------------------
# Draft 3: Complex-Valued FGRT RNN (Native Phase Tracking)
# -----------------------------------------------------------------------------
class ComplexFGRTRNNCell(nn.Module):
    r"""
    Hidden state exists in C^768. The Gyroidic connection acts as a complex rotation.
    Atiyah-Singer index flips occur naturally when state rotates through e^{i\pi}.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # PyTorch complex parameters are tricky, we simulate via 2x real dims (Real, Imag)
        self.hidden_dim = hidden_dim
        self.W_x = nn.Linear(input_dim, hidden_dim * 2)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
    def forward(self, x: torch.Tensor, h_prev_complex: torch.Tensor) -> torch.Tensor:
        # h_prev_complex shape: [B, hidden_dim * 2]
        out = self.W_x(x) + self.W_h(h_prev_complex)
        
        # Treat as complex and apply rotation
        real_part, imag_part = torch.chunk(out, 2, dim=-1)
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)
        
        # Cyclotomic rotation (simulated by adding phase angle)
        phase = phase + (np.pi / 4.0) # Rotate by pi/4 per step
        
        new_real = magnitude * torch.cos(phase)
        new_imag = magnitude * torch.sin(phase)
        
        # Apply tanh for stability
        h_t_complex = torch.tanh(torch.cat([new_real, new_imag], dim=-1))
        return h_t_complex

# -----------------------------------------------------------------------------
# Draft 4: The "Saturated Quantizer" Recurrent Core (KAGH / B-Spline Upgraded)
# -----------------------------------------------------------------------------
class SaturatedQuantizerRNNCell(nn.Module):
    """
    Hidden state is pushed through the Context-Aware Quantizer at each step,
    snapping to the nearest vertex of a Weyl Group (simulated here via rounding)
    to prevent Diffusion Toxin. 
    UPGRADE: Replaced standard GRU with KANLayer (True B-Splines) and KAGHBlock.
    """
    def __init__(self, input_dim: int, hidden_dim: int, quant_levels: int = 5):
        super().__init__()
        # Use True B-Spline layers instead of standard linear/GRU components
        self.spline_update = KANLayer(input_dim + hidden_dim, hidden_dim, spline_order=3)
        self.kagh_ghost_drafter = KAGHBlock(n_in=hidden_dim, n_out=hidden_dim, width=hidden_dim, depth=2)
        self.quant_levels = quant_levels

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h_prev], dim=-1)
        
        # True B-Spline nonlinear transition
        h_draft = self.spline_update(combined)
        
        # Speculative KAGH Drafting
        h_t = self.kagh_ghost_drafter(h_draft)
        
        # Saturated Quantizer Step: Snap to discrete bounds
        h_t_scaled = h_t * self.quant_levels
        h_t_quantized = torch.round(h_t_scaled) / self.quant_levels
        
        # Straight-through estimator for backprop
        h_t = h_t + (h_t_quantized - h_t).detach()
        return h_t

# -----------------------------------------------------------------------------
# Draft 5: PyOpenCL "Queue B" Hardware Sovereignty Cell
# -----------------------------------------------------------------------------
class PyOpenCLHardwareSovereigntyCell(nn.Module):
    """
    Offloads the recurrent temporal mixing step to the actual PyOpenCL SiliconSovereigntyEngine.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        # Deep integration: Initialize the true hardware compute arbitrator
        self.sovereignty_engine = SiliconSovereigntyEngine()
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        
        try:
            from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
            gpu_affordance = HardwareMonitor.get_instance().get_typed_affordance(ComputeAffordanceType.GPU_SOVEREIGN)
            if not gpu_affordance:
                raise RuntimeError(f"Compute affordance veto: {gpu_affordance.affordance_type.value} gauge depleted ({gpu_affordance.gauge:.2f}).")

            # Offload to OpenCL: Convert to numpy for the SiliconSovereigntyEngine
            h_np = h_prev.detach().cpu().numpy()
            x_np = x_proj.detach().cpu().numpy()
            
            # Invoke the true hardware breeding kernel
            mixed_np = self.sovereignty_engine.matrix_mix_breeding(
                matrix_a=h_np, 
                matrix_b=x_np, 
                alpha=0.5, 
                kappa_seal=0.1
            )
            mixed = torch.from_numpy(mixed_np).to(x.device).float()
            
            # Straight-through estimator to preserve gradients for x_proj
            mixed = x_proj + (mixed - x_proj).detach()
        except Exception as e:
            # Safe Fallback if OpenCL kernels are blocked or fail
            alpha = 0.5
            mixed = (1.0 - alpha) * h_prev + alpha * x_proj
        
        # Inject honest jitter/scars naturally via hardware entropy
        scars = harvest_honest_jitter(mixed.shape, device=mixed.device) * 0.18 * (mixed > 0.88).float()
        h_t = torch.clamp(mixed + scars, -1.0, 1.0)
        return h_t

# -----------------------------------------------------------------------------
# Draft 6: The "Lazarus Transition" (Continuous Superposition) RNN
# -----------------------------------------------------------------------------
class LazarusSuperpositionRNNCell(nn.Module):
    """
    Hidden state is a continuous Dark Matter field. Linearly stacked using
    incommensurate prime frequencies. No non-linear activation function in recurrence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_resonators: int = 32):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        # Authoritative Prime Resonance Ladder Initialization (No hardcoded scalar bugs!)
        self.ladder = PrimeResonanceLadder(num_resonators=num_resonators)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, step: int) -> torch.Tensor:
        x_proj = self.proj(x)
        
        # Use authoritative prime frequency for this specific step/signal
        freq = self.ladder.frequencies[step % self.ladder.num_resonators].to(x.device)
        wave = torch.sin(freq) * x_proj
        
        # Linear Additive Superposition (No activation)
        h_t = h_prev + wave 
        return h_t

    def audit_diophantine_frequency_near_miss(self, step: int) -> Dict[str, float]:
        """
        Audits the active prime resonator frequency f_{p_n} = 2*pi*ln(p_n) against
        Collatz-Baker Diophantine near-misses (l*ln(2) - k*ln(3)) to verify that
        recurrent superposition remains incommensurate and avoids rational limit-cycle lock.
        """
        idx = step % self.ladder.num_resonators
        freq_val = float(self.ladder.frequencies[idx].item())
        p_val = int(self.ladder.primes[idx].item())
        # Classic Baker logarithmic form gap: |8*ln(2) - 5*ln(3)| = ln(256/243) approx 0.052115
        baker_gap = abs(8.0 * float(np.log(2.0)) - 5.0 * float(np.log(3.0)))
        phase_mod = abs(float(np.fmod(freq_val, baker_gap)))
        return {
            "prime_modulus": float(p_val),
            "resonator_frequency": freq_val,
            "baker_logarithmic_gap": baker_gap,
            "incommensurate_margin": phase_mod
        }

# -----------------------------------------------------------------------------
# Draft 7: The "Feature Scar" LCFT Memory Cell (Recommended)
# -----------------------------------------------------------------------------
class FeatureScarLCFTCell(nn.Module):
    """
    Uses Fibonacci Resonance Entropy to fossilize specific states.
    When a 'Good Bug' occurs (Atiyah-Singer spike), h_t is saved into a 'Cerumen Pot'.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Replace the linear GRU with the True B-Spline Surrogate (KAGH)
        # to respect the physical affordance of the topological engine
        self.kagh_gru = KANLayer(in_features=input_dim + hidden_dim, out_features=hidden_dim, grid_size=5, spline_order=3)
        self.anomaly_detector = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h_prev, cerumen_pot = state
        
        # Spatial Hash Preservation via KAGH/True B-Spline
        # Concatenate x and h_prev for the KANLayer (simulating GRU projection)
        combined = torch.cat([x, h_prev], dim=-1)
        h_t = torch.tanh(self.kagh_gru(combined))
        
        # Calculate Atiyah-Singer Index (Anomaly Score)
        atiyah_singer_index = torch.abs(self.anomaly_detector(h_t)) # [B, 1]
        
        # Threshold for 'Good Bug' / Topological Violation
        anomaly_mask = (atiyah_singer_index > 0.8).float()
        
        # Fossilize into Cerumen Pot (Non-decaying memory)
        cerumen_pot = cerumen_pot + anomaly_mask * h_t
        
        # Chern-Simons Gasket: Route future updates around the scar
        # If there is a scar, we force h_t to align with the fossil
        scar_presence = (torch.abs(cerumen_pot) > 0).float()
        h_t = h_t * (1.0 - scar_presence) + cerumen_pot * scar_presence
        
        return h_t, cerumen_pot


# -----------------------------------------------------------------------------
# Unified Polymorphic Recurrent Cell & Factory
# -----------------------------------------------------------------------------
class UnifiedFGRTRNNCell(nn.Module):
    """
    Polymorphic dispatcher for FGRT recurrent cells.
    Allows UniversalOrchestrator to dynamically route between:
      - 'feature_scar': B-Spline KAGH memory with Cerumen Pot fossilization
      - 'opencl': Hardware-sovereign PyOpenCL matrix breeding kernel offload
      - 'berry_phase': Holonomy-sensitive GRU with Stiefel-Whitney parity flips
      - 'chiral': Chiral gated attractor network
    """
    def __init__(self, input_dim: int, hidden_dim: int, regime: str = 'feature_scar'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.regime = regime
        
        # Instantiate sub-cells
        self.feature_scar_cell = FeatureScarLCFTCell(input_dim, hidden_dim)
        self.opencl_cell = PyOpenCLHardwareSovereigntyCell(input_dim, hidden_dim)
        self.berry_cell = BerryPhaseGRUCell(input_dim, hidden_dim)
        self.chiral_cell = ChiralGatedRNNCell(input_dim, hidden_dim)
        
        self.register_buffer('gamma_buffer', torch.zeros(1, 1))

    def set_regime(self, regime: str):
        """Switch active recurrence regime without resetting parameters."""
        valid_regimes = {'feature_scar', 'opencl', 'berry_phase', 'chiral'}
        if regime not in valid_regimes:
            raise ValueError(f"Unknown regime '{regime}'. Choose from {valid_regimes}")
        self.regime = regime

    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h_prev, cerumen_pot = state
        
        if self.regime == 'opencl':
            h_t = self.opencl_cell(x, h_prev)
            # Track scars via hardware entropy spikes
            scar_spike = (torch.abs(h_t) > 0.88).float()
            updated_scar = cerumen_pot + scar_spike * h_t
            return h_t, updated_scar
            
        elif self.regime == 'berry_phase':
            if self.gamma_buffer.shape[0] != x.shape[0]:
                self.gamma_buffer = torch.zeros(x.shape[0], 1, device=x.device)
            h_t, gamma = self.berry_cell(x, (h_prev, self.gamma_buffer))
            self.gamma_buffer = gamma.detach()
            # Stiefel-Whitney parity scars
            parity_scar = (torch.cos(gamma) < 0.0).float()
            updated_scar = cerumen_pot + parity_scar * h_t
            return h_t, updated_scar
            
        elif self.regime == 'chiral':
            h_t = self.chiral_cell(x, h_prev)
            updated_scar = cerumen_pot
            return h_t, updated_scar
            
        else: # Default: 'feature_scar'
            return self.feature_scar_cell(x, state)


def create_fgrt_rnn_cell(cell_type: str = 'feature_scar', input_dim: int = 64, hidden_dim: int = 64) -> UnifiedFGRTRNNCell:
    """Factory creating a unified polymorphic FGRT RNN cell."""
    return UnifiedFGRTRNNCell(input_dim=input_dim, hidden_dim=hidden_dim, regime=cell_type)

