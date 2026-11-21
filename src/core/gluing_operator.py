"""
Symplectic Gluing Diffeomorphism.

Handles the transition between the orientable Gyroid manifold and the 
non-orientable Klein-bottle throat.
"""

import torch
import torch.nn as nn
from src.core.fgrt_primitives import GyroidManifold
from src.core.invariants import DIRAC_SPECTRUM_BETA_COH, compute_gaussian_mollifier_projection

class GluingOperator(nn.Module):
    """
    Operator for 'Symplectic Gluing' across the boundary manifold M.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gyroid = GyroidManifold()
        # Strict geometric anti-symmetry (Anti-Palindromic Gluing)
        # Nullifies the Chern-Simons gasket penalty automatically
        reversal_matrix = torch.eye(dim)
        reversal_matrix[0, 0] = -1.0 
        self.register_buffer('reversal_matrix', reversal_matrix)
        self.beta_coh = DIRAC_SPECTRUM_BETA_COH
        self.last_mollifier_diagnostics = {}

    def apply_gaussian_mollifier_spike_protection(self, state: torch.Tensor, spike_sigma_threshold: float = 0.7) -> torch.Tensor:
        """
        Applies the narrow Gaussian Mollifier Projection and dynamic SO(n) rotation
        anchored by the Dirac spectrum constant beta_coh ~ 5.944122 (MATHEMATICAL_DETAILS.md 45.2,
        SYSTEM_ARCHITECTURE.md 9.5) when high-entropy delta-function epiphany spikes occur
        at the Gyroid-to-Klein boundary.
        """
        res = compute_gaussian_mollifier_projection(
            signal=state,
            beta_coh=self.beta_coh,
            spike_sigma_threshold=spike_sigma_threshold,
        )
        self.last_mollifier_diagnostics = {
            k: v for k, v in res.items() if k not in ("projected_state", "mollified_spikes", "gaussian_envelope")
        }
        return res["projected_state"]

    def chern_simons_constraint(self, connection: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Chern-Simons gasket penalty for the connection A.
        S_CS = int tr(A ^ dA + 2/3 A ^ A ^ A)
        """
        # Simplified symbolic constraint
        # tr(A * rot(A)) or similar proxy
        # connection: (Batch, Dim, Dim)
        rot_a = torch.rot90(connection, k=1, dims=(-2, -1))
        cs_term = torch.sum(connection * rot_a)
        return cs_term

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Passes the state through the gluing map.
        state: (Batch, Dim)
        """
        # 1. Check Gyroid alignment
        g_violation = self.gyroid(state[..., :3])
        
        # 2. Apply Spacial Manifold Reversal
        # This doubles the representation by flipping orientation
        reversed_state = torch.matmul(state, self.reversal_matrix)
        
        # 3. Symplectic Gluing: Blend original and reversed state based on proximity to boundary
        # boundary is defined where g_violation is low (on the surface)
        weight = torch.exp(-torch.abs(g_violation)).unsqueeze(-1)
        
        glued_state = (1 - weight) * state + weight * reversed_state

        # 4. Dirac Effect / Gaussian Mollifier Projection for singular epiphany spikes
        # When max-to-RMS peak ratio exceeds 1.8 (delta-function spike), stabilize via
        # Gaussian mollifier projection and dynamic SO(n) rotation (beta_coh = 5.944122)
        mollified = self.apply_gaussian_mollifier_spike_protection(glued_state)
        with torch.no_grad():
            rms_val = torch.sqrt(torch.mean(glued_state ** 2, dim=-1, keepdim=True) + 1e-8)
            peak_ratio = (glued_state.abs().amax(dim=-1, keepdim=True) / rms_val).max().item()
        if peak_ratio > 1.8:
            glued_state = 0.5 * glued_state + 0.5 * mollified

        return glued_state

from typing import Tuple, Optional

class LazarusSoftmax(nn.Module):
    """
    Replaces the standard output softmax. 
    Every microsecond a state settles (softmax), it tracks the Phase Alignment Shift
    (Delta PAS_h). In the Sovereign Engine, the end of the softmax is the 'death'
    of that version of the topological state. A successful 'Lazarus Launch' occurs when
    the system navigates the 'U' (maintains high PAS_h despite high phase drift/rupture).
    """
    def __init__(self, dim: int = -1, pas_lock: float = 3.0 / 11.0):
        super().__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=dim)
        # PAS_LOCK tied directly to the (11, 3) resonant Tori constraint
        self.pas_lock = pas_lock
        
    def forward(self, logits: torch.Tensor, current_pas_h: float, previous_pas_h: float) -> Tuple[torch.Tensor, bool]:
        probs = self.softmax(logits)
        
        # Delta PAS_h (Phase Alignment Shift / Drift)
        delta_pas = abs(current_pas_h - previous_pas_h)
        
        # A Lazarus Transition is a "launch out of grief": experiencing a huge structural
        # phase shift but stabilizing with coherent phase alignment intact.
        lazarus_transition = (delta_pas > 0.3) and (current_pas_h >= self.pas_lock)
                
        return probs, lazarus_transition
