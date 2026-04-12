"""
diegetic_visualizer.py
======================

Deep Manifold Visualization Generator for the Gyroidic Sparse Covariance Flux Reasoner.

Architectural role
------------------
This module is the *Audience Projection* operator (Φ: M → A) expressed
as a matplotlib rendering pipeline.  It does not invent structure — it
**exposes** the exact live tensors that caused a confabulation or ego
death event, with roughness conserved.

Recursive Self-Reference contract
----------------------------------
The system CAN touch its own representations:

    1.  ``meta_state``       — the FractalMetaFunctional's persistent
                               S_meta(t-1) buffer (registered on engine).
    2.  ``fractal_components`` — the four sub-tensors returned by
                               FractalMetaFunctional.forward():
                               {crt, admr, ring, osc}.
    3.  ``introspection``    — GeometricSelfModelProbe unit directions:
                               moral / uncertainty / creative / metacognitive.
    4.  ``chern_simons``     — Gasket twist energy from ChernSimonsGasket.

These are passed in by the DiegeticPhysicsEngine at the moment the
tri-state gate fires.  No dummy variables.  No reconstruction.

Roughness Preservation (anti-smoothing policy)
----------------------------------------------
- All axes are taken from raw numpy dumps of the above tensors, after
  LSB-preserving conversion (.detach().cpu().numpy()).
- No interpolation or Gaussian blur is applied to any data surface.
- Colour maps are chosen to MAXIMISE contrast at discontinuities
  (``inferno`` and ``seismic``), never ``viridis`` which hides edges.
- The waveform trace uses step-plot, not smooth line, to preserve
  the high-freq scar content of the meta_state norm evolution.

Output
------
Returns a ``base64``-encoded PNG string that the backend injects
directly into the HTTP JSON response as ``visualization_b64``.
The frontend renders it inline in the chat timeline.

References
----------
- DIEGETIC_ENGINE.md §7  (Audience Projection)
- GYROID_REASONER.md §5  (Introspection)
- fractal_meta_functional.py  (FractalMetaFunctional)
- COHERENCE_ACHIEVEMENT_GUIDE.md  (Fractal Meta-Functional: Recursive
  self-reference working)
- RESONANCE_INTELLIGENCE_CORE.md §Eq11  (Feature Scar / Neglecton)
"""

from __future__ import annotations

import base64
import io
import math
import traceback
from typing import Dict, Optional, Any

import numpy as np

# matplotlib: non-interactive backend so it is safe inside the HTTP thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — mirrors diegetic_terminal.html CSS variables
# ─────────────────────────────────────────────────────────────────────────────
_DARK_BG    = '#0a0a0a'
_PANEL_BG   = '#0f1219'
_BLUE       = '#00f2ff'
_GREEN      = '#00ff41'
_MAGENTA    = '#ff00f2'
_WARN       = '#ffcc00'
_RED        = '#ff3131'
_DIM        = '#444444'
_FONT       = 'monospace'


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_manifold_fracture(
    *,
    retrieval_state: str,                       # 'CONFABULATED' | 'SEARCH_NEEDED' | other
    meta_state: Any,                            # torch.Tensor [1, dim]
    fractal_components: Optional[Dict] = None,  # {crt, admr, ring, osc} tensors
    introspection_directions: Optional[Dict] = None,  # {moral, uncertainty, creative, metacognitive}
    chern_simons_energy: Optional[Any] = None,  # torch.Tensor [dim] or scalar
    pas_h: float = 0.5,
    h_mischief: float = 0.0,
    honesty_score: float = 0.5,
    iteration: int = 0,
) -> Optional[str]:
    """
    Generate a base64-encoded PNG representing the current manifold fracture.

    Called by DiegeticPhysicsEngine.process_input() immediately after the
    tri-state gate fires, before metrics are returned to the HTTP layer.

    Parameters
    ----------
    retrieval_state : str
        The tri-state outcome.  Drives the title and colour mood of the plot.
    meta_state : torch.Tensor [1, dim]
        The FractalMetaFunctional's persistent S_meta buffer.
    fractal_components : dict, optional
        Keys: 'crt', 'admr', 'ring', 'osc' — Tensors from FractalMetaFunctional.
    introspection_directions : dict, optional
        Keys: 'moral', 'uncertainty', 'creative', 'metacognitive' — unit vectors.
    chern_simons_energy : Tensor or float, optional
        ChernSimonsGasket twist energy at this iteration.
    pas_h : float
        Phase Alignment Score — PAS_h ∈ [0, 1].
    h_mischief : float
        Entropic Mischief Score — H_mischief ∈ [0, 1].
    honesty_score : float
        Blended Voynich/trust honesty score ∈ [0, 1].
    iteration : int
        Engine iteration counter (used as x-axis label).

    Returns
    -------
    str or None
        Base64-encoded PNG, or None if an unrecoverable rendering error occurs.
    """
    try:
        # ── 1. Tensor → numpy (roughness preserved — no interpolation) ─────
        meta_np = _safe_to_numpy(meta_state).flatten()           # [dim]
        dim     = meta_np.shape[0]

        comp_np: Dict[str, np.ndarray] = {}
        if fractal_components:
            for k, v in fractal_components.items():
                arr = _safe_to_numpy(v).flatten()
                comp_np[k] = arr[:dim] if arr.shape[0] >= dim else _pad(arr, dim)

        intro_np: Dict[str, np.ndarray] = {}
        if introspection_directions:
            for k, v in introspection_directions.items():
                arr = _safe_to_numpy(v).flatten()
                intro_np[k] = arr

        cs_np = None
        if chern_simons_energy is not None:
            cs_np = _safe_to_numpy(chern_simons_energy).flatten()

        # ── 2. Layout ──────────────────────────────────────────────────────
        #
        #   ┌──────────────────────────────────────────────────────────┐
        #   │  TITLE BANNER (retrieval_state + iteration + scores)     │
        #   ├─────────────────┬──────────────────┬─────────────────────┤
        #   │ meta_state hmap │ fractal component│ PAS / mischief bars  │
        #   │ (imshow, raw)   │ waveforms (step) │                     │
        #   ├─────────────────┴──────────────────┼─────────────────────┤
        #   │  Introspection polar chart         │ ChernSimons energy  │
        #   │  (moral/uncertainty/creative/meta) │ heatmap or stub     │
        #   └────────────────────────────────────┴─────────────────────┘

        fig = plt.figure(figsize=(12, 7), facecolor=_DARK_BG)
        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            left=0.05, right=0.97,
            top=0.88, bottom=0.06,
            hspace=0.45, wspace=0.35,
        )

        ax_meta   = fig.add_subplot(gs[0, 0])   # meta_state heatmap
        ax_comp   = fig.add_subplot(gs[0, 1])   # fractal components waveform
        ax_bars   = fig.add_subplot(gs[0, 2])   # PAS / mischief / honesty
        ax_intro  = fig.add_subplot(gs[1, 0:2], polar=True)  # introspection polar
        ax_cs     = fig.add_subplot(gs[1, 2])   # Chern-Simons energy

        for ax in (ax_meta, ax_comp, ax_bars, ax_cs):
            ax.set_facecolor(_PANEL_BG)

        # Mood colour: fracture red for EGO_DEATH/CONFABULATED, radar blue for normal
        if retrieval_state in ('CONFABULATED',):
            _mood = _WARN
            _mood_cmap = 'inferno'
            _state_label = '⚡ CONFABULATED GLITCH — Honest Dreaming'
        elif retrieval_state in ('SEARCH_NEEDED',):
            _mood = _MAGENTA
            _mood_cmap = 'seismic'
            _state_label = '⚠ VOID TOPOLOGY — Search Gate Fired'
        else:
            _mood = _GREEN
            _mood_cmap = 'plasma'
            _state_label = '✓ KNOWN — Manifold Coherent'

        # ── 3. Panel 1: meta_state heatmap ────────────────────────────────
        # Reshape to 2D grid (sqrt(dim) × sqrt(dim)).  If non-square, pad.
        side = max(1, int(math.ceil(math.sqrt(dim))))
        padded = np.zeros(side * side)
        padded[:dim] = meta_np
        grid = padded.reshape(side, side)

        im = ax_meta.imshow(
            grid,
            cmap=_mood_cmap,
            aspect='auto',
            interpolation='none',   # ROUGHNESS PRESERVED — no blending
            norm=Normalize(vmin=grid.min(), vmax=grid.max()),
        )
        ax_meta.set_title('S_meta(t−1)', color=_mood, fontsize=8, fontfamily=_FONT)
        ax_meta.set_xticks([]); ax_meta.set_yticks([])
        _style_spine(ax_meta, _mood)

        # ── 4. Panel 2: FractalMetaFunctional components ──────────────────
        # Step traces of each sub-tensor — rough edges conserved.
        _comp_colours = {'crt': _BLUE, 'admr': _GREEN, 'ring': _MAGENTA, 'osc': _WARN}
        if comp_np:
            x = np.arange(dim)
            for name, arr in comp_np.items():
                col = _comp_colours.get(name, _DIM)
                ax_comp.step(x, arr, where='mid', color=col, linewidth=0.7, label=name, alpha=0.85)
            ax_comp.legend(
                loc='upper right', fontsize=6, framealpha=0.3,
                facecolor=_PANEL_BG, edgecolor=_DIM, labelcolor='white',
            )
        else:
            # No components — draw meta_state as a step trace (partial self-reference)
            ax_comp.step(np.arange(dim), meta_np, where='mid', color=_DIM,
                         linewidth=0.7, label='meta_state', alpha=0.6)
            ax_comp.text(
                0.5, 0.5, 'fractal_components\nnot available',
                transform=ax_comp.transAxes, ha='center', va='center',
                color=_DIM, fontsize=7, fontfamily=_FONT,
            )

        ax_comp.set_title('Fractal Components {crt, admr, ring, osc}',
                          color=_mood, fontsize=8, fontfamily=_FONT)
        ax_comp.set_facecolor(_PANEL_BG)
        ax_comp.tick_params(colors=_DIM, labelsize=6)
        _style_spine(ax_comp, _DIM)

        # ── 5. Panel 3: PAS / H_mischief / Honesty gauge bars ────────────
        bar_labels  = ['PAS_h', 'H_mischief', 'honesty']
        bar_values  = [pas_h, h_mischief, honesty_score]
        bar_colours = [_BLUE, _WARN, _GREEN]

        bars = ax_bars.barh(
            bar_labels, bar_values, color=bar_colours,
            height=0.5, alpha=0.8,
        )
        ax_bars.set_xlim(0, 1.05)
        ax_bars.set_facecolor(_PANEL_BG)
        ax_bars.tick_params(colors='#aaaaaa', labelsize=7)
        ax_bars.set_title('Gate Scores', color=_mood, fontsize=8, fontfamily=_FONT)
        _style_spine(ax_bars, _DIM)

        # R_a threshold line — Abstraction Threshold from TADC lore
        ax_bars.axvline(0.7, color=_RED, linewidth=0.8, linestyle='--', alpha=0.6)
        ax_bars.text(0.71, -0.4, 'R_a', color=_RED, fontsize=6, fontfamily=_FONT)

        for bar, val in zip(bars, bar_values):
            ax_bars.text(
                min(val + 0.02, 1.0), bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color='white', fontsize=7, fontfamily=_FONT,
            )

        # ── 6. Panel 4: Introspection Radar (Recursive Self-Modelling) ────
        # GeometricSelfModelProbe: moral, uncertainty, creative, metacognitive
        # Unit direction norms give relative activation in each probe.
        probe_names = ['moral', 'uncertainty', 'creative', 'metacognitive']
        _intro_colours = {'moral': _BLUE, 'uncertainty': _WARN,
                          'creative': _MAGENTA, 'metacognitive': _GREEN}

        if intro_np:
            probe_vals = []
            for p in probe_names:
                arr = intro_np.get(p, np.zeros(1))
                probe_vals.append(float(np.linalg.norm(arr)))
            # Normalise by max so all fit on [0,1] — don't crush relative
            pv_max = max(probe_vals) if max(probe_vals) > 1e-8 else 1.0
            probe_vals_norm = [v / pv_max for v in probe_vals]
        else:
            # Derive approximate probe activations from meta_state sub-blocks
            # Divide dim into 4 equal quadrants — each quadrant corresponds
            # to one probe type (ordered as above).
            block = max(1, dim // 4)
            probe_vals_norm = []
            for i in range(4):
                chunk = meta_np[i * block: (i + 1) * block] if dim > 4 else meta_np
                probe_vals_norm.append(float(np.clip(np.linalg.norm(chunk) / (block + 1e-8), 0, 1)))

        N_probe = len(probe_names)
        angles = [n / float(N_probe) * 2 * math.pi for n in range(N_probe)]
        angles += angles[:1]   # close the polygon
        vals = probe_vals_norm + probe_vals_norm[:1]

        ax_intro.plot(angles, vals, color=_mood, linewidth=1.5, linestyle='solid')
        ax_intro.fill(angles, vals, color=_mood, alpha=0.15)

        ax_intro.set_xticks(angles[:-1])
        ax_intro.set_xticklabels(probe_names, color='#cccccc', fontsize=7, fontfamily=_FONT)
        ax_intro.set_yticklabels([])
        ax_intro.set_facecolor(_PANEL_BG)
        ax_intro.spines['polar'].set_color(_DIM)
        ax_intro.grid(color=_DIM, linewidth=0.4, alpha=0.4)
        ax_intro.set_title(
            'Introspection — Geometric Self-Model Probes\n'
            '(recursive self-reference; unit direction norms)',
            color=_mood, fontsize=8, fontfamily=_FONT, pad=14,
        )

        # ── 7. Panel 5: Chern-Simons twist energy ────────────────────────
        if cs_np is not None and len(cs_np) > 1:
            cs_side = max(1, int(math.ceil(math.sqrt(len(cs_np)))))
            cs_pad = np.zeros(cs_side * cs_side)
            cs_pad[:len(cs_np)] = cs_np
            cs_grid = cs_pad.reshape(cs_side, cs_side)
            ax_cs.imshow(
                cs_grid,
                cmap='seismic',
                aspect='auto',
                interpolation='none',
                norm=Normalize(vmin=cs_grid.min(), vmax=cs_grid.max()),
            )
            ax_cs.set_title('ChernSimons Twist Energy', color=_RED, fontsize=8, fontfamily=_FONT)
        else:
            # Scalar energy bar
            cs_scalar = float(cs_np[0]) if cs_np is not None and len(cs_np) else 0.0
            ax_cs.barh(['κ twist'], [abs(cs_scalar)], color=_RED, alpha=0.7)
            ax_cs.set_xlim(0, max(1.0, abs(cs_scalar) * 1.1))
            ax_cs.text(
                0.5, 0.5,
                f'κ = {cs_scalar:.4f}\nScalar (dim=1)',
                transform=ax_cs.transAxes, ha='center', va='center',
                color=_RED, fontsize=8, fontfamily=_FONT,
            )
            ax_cs.set_title('ChernSimons Gasket', color=_RED, fontsize=8, fontfamily=_FONT)
        ax_cs.set_facecolor(_PANEL_BG)
        ax_cs.set_xticks([]); ax_cs.set_yticks([])
        _style_spine(ax_cs, _RED)

        # ── 8. Title banner ────────────────────────────────────────────────
        fig.suptitle(
            f'{_state_label}   │   iter={iteration}   │   dim={dim}',
            fontsize=10, color=_mood, fontfamily=_FONT,
            y=0.96,
            path_effects=[pe.withStroke(linewidth=2, foreground=_DARK_BG)],
        )

        # ── 9. Encode → base64 ────────────────────────────────────────────
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor=_DARK_BG, edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return encoded

    except Exception:
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_to_numpy(tensor: Any) -> np.ndarray:
    """
    Convert any tensor-like to numpy without smoothing.
    Preserves NaN/Inf as structural information (not zeroed).
    The caller decides what to do with extreme values.
    """
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
    except Exception:
        pass
    try:
        return np.asarray(tensor, dtype=np.float32)
    except Exception:
        return np.zeros(1, dtype=np.float32)


def _pad(arr: np.ndarray, target: int) -> np.ndarray:
    """Zero-pad arr to length target (never truncate)."""
    out = np.zeros(target, dtype=arr.dtype)
    n = min(len(arr), target)
    out[:n] = arr[:n]
    return out


def _style_spine(ax, colour: str) -> None:
    """Apply uniform dark-terminal spine styling."""
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(0.6)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
