"""
Context-Aware Quantizer (CAQ).

Implements per-axis, context-sensitive quantization step sizes on top of
MetaPolytopeMatrioshka, realising the core quantized evolution equation:

    x_{t+1} = Q_{Z_t}(F(Q_{Z_t}(x_t)))

from ai project report_2-2-2026.txt §3 "Matrioshka Quantized Windows".

Design: 
    - delta_log: [max_depth+1, dim] learnable log-space per-axis step sizes.
      Depth 0 is the outermost shell (coarse), depth max_depth is the innermost (fine).
    - PAS anisotropy: Phase Alignment Scores from CALM modulate per-axis steps.
      High-coherence axes get finer quantization; low-coherence stays coarse.
    - Carries (alpha, level) state between calls so the quantizer "knows" which
      Matrioshka shell it is currently inhabiting.
    - Emits BoundaryState on shell crossing (level change) for downstream veto logic.

References:
    - ai project report_2-2-2026.txt §3 (CAQ, Asymptotic Windowing)
    - VETO_SUBSPACE_ARCHITECTURE.md §5 (BoundaryState)
    - ADVANCED_MATHEMATICAL_EXTENSIONS.md §1 (Meta-Polytope Matrioshka)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.core.meta_polytope_matrioshka import MetaPolytopeMatrioshka, BoundaryState


class ContextAwareQuantizer(nn.Module):
    """
    Context-Aware Quantizer with per-axis, depth-adaptive step sizes.

    The quantization step applied to axis j at shell depth ℓ is:

        Δ_{jℓ} = exp(delta_log[ℓ, j]) * base_step / 2^ℓ

    PAS anisotropy further modulates Δ:

        Δ̃_{jℓ} = Δ_{jℓ} / (1 + pas_anisotropy * PAS_j)

    where PAS_j ∈ [0,1] is the phase-alignment score on axis j.
    High PAS → smaller step (finer resolution on coherent directions).
    Low PAS  → larger step (coarse quantization on incoherent directions).
    """

    def __init__(
        self,
        dim: int,
        max_depth: int = 5,
        base_step: float = 0.1,
        pas_anisotropy: float = 2.0,
    ):
        """
        Args:
            dim:            State vector dimensionality.
            max_depth:      Number of Matrioshka shells.
            base_step:      Outermost shell base step size (before log warp).
            pas_anisotropy: Scaling factor for PAS-based anisotropy.
        """
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.base_step = base_step
        self.pas_anisotropy = pas_anisotropy

        # Per-axis, per-depth step parameters (log-space → always positive)
        # Shape: [max_depth+1, dim]. Initialised to zero ≡ step = base_step / 2^ℓ.
        self.delta_log = nn.Parameter(torch.zeros(max_depth + 1, dim))

        # Underlying Matrioshka quantizer (handles CRT shell switching)
        self.matrioshka = MetaPolytopeMatrioshka(max_depth=max_depth, base_dim=dim)

        # Persistent shell state: carried between forward() calls.
        # Registered as plain Python ints (not buffers) — they're control flow,
        # not model weights, so they don't need gradient tracking.
        self._alpha: int = 0
        self._level: int = 0

        # Diagnostics cache
        self._last_diag: Dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_step_sizes(
        self,
        pas_scores: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the effective per-axis step tensor Δ̃ for the current level.

        Returns:
            step: [1, dim] positive step sizes for element-wise quantisation.
        """
        ℓ = min(self._level, self.max_depth)

        # Base step shrinks geometrically with depth
        depth_scale = self.base_step / (2.0 ** ℓ)

        # Learnable log-warp for this depth level
        delta = torch.exp(self.delta_log[ℓ]) * depth_scale  # [dim]

        # PAS anisotropy: finer resolution on coherent axes
        if pas_scores is not None:
            # pas_scores: [dim] scores in [0, 1]
            pas = pas_scores.to(device).view(-1)[: self.dim]
            if pas.shape[0] < self.dim:
                # Pad with zeros (unknown coherence → no anisotropy correction)
                pad = torch.zeros(self.dim - pas.shape[0], device=device)
                pas = torch.cat([pas, pad])
            # Shrink step proportional to coherence
            aniso = 1.0 + self.pas_anisotropy * pas
            delta = delta / aniso

        return delta.unsqueeze(0)  # [1, dim]

    # ------------------------------------------------------------------
    # Main quantisation pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _quantize(
        self, x: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        """Round x to the nearest multiple of step (per-axis)."""
        # Avoid division by near-zero
        safe_step = step.clamp(min=1e-6)
        return torch.round(x / safe_step) * safe_step

    def forward(
        self,
        x: torch.Tensor,
        pas_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[BoundaryState]]:
        """
        Apply context-aware quantization.

        Args:
            x:          [batch, dim] state vector.
            pas_scores: Optional [dim] per-axis Phase Alignment Scores in [0,1].
                        If None, uniform (no anisotropy) is assumed.

        Returns:
            q_out:          [batch, dim] quantised state.
            boundary_state: BoundaryState if a shell crossing occurred, else None.
        """
        device = x.device
        prev_level = self._level

        # 1. Compute per-axis step sizes
        step = self._compute_step_sizes(pas_scores, device)  # [1, dim]

        # 2. Quantise with per-axis steps
        q = self._quantize(x, step)  # [batch, dim]

        # 3. Hand off to Matrioshka for CRT-level shell transition
        #    matrioshka.forward returns (quantized, new_alpha, new_level)
        #    but uses a *uniform* step internally; we have already done the
        #    per-axis step, so we pass q (already quantised) and ignore its
        #    quantisation result — we only want the shell transition logic.
        with torch.no_grad():
            _, self._alpha, self._level = self.matrioshka(
                q, alpha=self._alpha, level=self._level
            )

        # 4. Detect shell crossing → emit BoundaryState
        boundary: Optional[BoundaryState] = None
        if self._level != prev_level:
            # Compute facet normal proxy: direction of the quantisation error
            error = x - q  # [batch, dim]
            direction = error.mean(dim=0)  # [dim]
            d_norm = direction / (torch.norm(direction) + 1e-8)
            # Facet normal: unit vector in the changed dimension
            facet_n = torch.zeros_like(d_norm)
            facet_n[torch.argmax(torch.abs(d_norm))] = 1.0
            boundary = BoundaryState.from_crossing(
                state_direction=d_norm,
                facet_normal=facet_n,
                alpha=self._alpha,
                level=self._level,
                max_level=self.max_depth,
            )

        # 5. Update diagnostics
        self._last_diag = {
            "level": self._level,
            "alpha": self._alpha,
            "step_mean": step.mean().item(),
            "step_min": step.min().item(),
            "step_max": step.max().item(),
            "shell_crossed": boundary is not None,
            "quant_error_rms": (x - q).pow(2).mean().sqrt().item(),
        }

        return q, boundary

    def reset_shell_state(self) -> None:
        """Reset to outermost shell. Call at session start or after critical veto."""
        self._alpha = 0
        self._level = 0

    def get_diagnostics(self) -> Dict:
        """Return diagnostics from the most recent forward() call."""
        return dict(self._last_diag)
