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

Return contract
---------------
``render_manifold_fracture()`` returns a **dict** (not a bare string):

    {
        "b64": str | None,                   # base64-encoded PNG
        "structural_residues": list[float],  # LSB-rounded probe+component norms
        "cheby_self_fingerprint": list[float], # Chebyshev of PNG luminance
        "betti_matrix": list | None,         # [beta0, beta1] if CONFABULATED
    }

The backend injects these directly into meta_state (Introspection κ·I channel)
and reads b64 for the HTTP response.

Roughness Preservation (anti-smoothing policy)
----------------------------------------------
- No interpolation or Gaussian blur.
- Colour maps ``inferno`` and ``seismic`` maximise discontinuity contrast.
- Step-plot (not line) for component waveforms.

References
----------
DIEGETIC_ENGINE.md §7, GYROID_REASONER.md §5,
fractal_meta_functional.py, RESONANCE_INTELLIGENCE_CORE.md §11
"""

from __future__ import annotations

import base64
import io
import math
import struct
import traceback
from typing import Dict, Optional, Any, List

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
_DARK_BG  = '#0a0a0a'
_PANEL_BG = '#0f1219'
_BLUE     = '#00f2ff'
_GREEN    = '#00ff41'
_MAGENTA  = '#ff00f2'
_WARN     = '#ffcc00'
_RED      = '#ff3131'
_DIM      = '#444444'
_FONT     = 'monospace'

# ─────────────────────────────────────────────────────────────────────────────
# K for the self-Chebyshev decomposition (derived from PNG pixel count below)
# ─────────────────────────────────────────────────────────────────────────────
_K_SELF_FP_MAX = 32


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_manifold_fracture(
    *,
    retrieval_state: str,
    meta_state: Any,
    fractal_components: Optional[Dict] = None,
    introspection_directions: Optional[Dict] = None,
    chern_simons_energy: Optional[Any] = None,
    pas_h: float = 0.5,
    h_mischief: float = 0.0,
    honesty_score: float = 0.5,
    iteration: int = 0,
) -> Dict[str, Any]:
    """
    Generate a base64-encoded PNG, extract structural residues and a
    Chebyshev self-fingerprint from the result, and (on CONFABULATED)
    compute a Betti barcode.

    Returns
    -------
    dict with keys:
        b64                   : str | None  — base64 PNG
        structural_residues   : list[float] — LSB-rounded norms
        cheby_self_fingerprint: list[float] — Chebyshev of PNG luminance
        betti                 : dict | None — {beta0, beta1} if CONFABULATED
    """
    result: Dict[str, Any] = {
        "b64": None,
        "structural_residues": [],
        "cheby_self_fingerprint": [],
        "betti": None,
    }

    try:
        # ── 1. Tensor → numpy ──────────────────────────────────────────────
        meta_np = _safe_to_numpy(meta_state).flatten()
        dim = meta_np.shape[0]

        comp_np: Dict[str, np.ndarray] = {}
        if fractal_components:
            for k, v in fractal_components.items():
                arr = _safe_to_numpy(v).flatten()
                comp_np[k] = arr[:dim] if arr.shape[0] >= dim else _pad(arr, dim)

        intro_np: Dict[str, np.ndarray] = {}
        if introspection_directions:
            for k, v in introspection_directions.items():
                intro_np[k] = _safe_to_numpy(v).flatten()

        cs_np = None
        if chern_simons_energy is not None:
            cs_np = _safe_to_numpy(chern_simons_energy).flatten()

        # ── 2. Structural residues (probe norms + component norms) ─────────
        probe_names = ['moral', 'uncertainty', 'creative', 'metacognitive']
        block = max(1, dim // 4)
        probe_vals = []
        for i, p in enumerate(probe_names):
            if p in intro_np:
                probe_vals.append(float(np.linalg.norm(intro_np[p])))
            else:
                chunk = meta_np[i * block: (i + 1) * block]
                probe_vals.append(float(np.linalg.norm(chunk) / (len(chunk) + 1e-8)))

        comp_norms = [float(np.linalg.norm(v)) for v in comp_np.values()]
        raw_residues = np.array(probe_vals + comp_norms, dtype=np.float32)
        result["structural_residues"] = _lsb_round_np(raw_residues, scale=1024.0).tolist()

        # ── 3. Introspection polar values (normalised) ─────────────────────
        pv_max = max(probe_vals) if max(probe_vals) > 1e-8 else 1.0
        probe_vals_norm = [v / pv_max for v in probe_vals]

        # ── 4. Betti computation on CONFABULATED ───────────────────────────
        betti_result = None
        if retrieval_state == 'CONFABULATED' and comp_np:
            try:
                from src.core.quantum_tda import QuantumBettiApproximator
                # Build adjacency from pairwise cosine distances of component vectors
                comps = list(comp_np.values())
                N = len(comps)
                adj = np.zeros((N, N), dtype=np.float32)
                for i in range(N):
                    for j in range(i + 1, N):
                        a, b_ = comps[i], comps[j]
                        cos = float(np.dot(a, b_) / (np.linalg.norm(a) * np.linalg.norm(b_) + 1e-8))
                        adj[i, j] = adj[j, i] = max(0.0, cos)
                import torch as _t
                adj_t = _t.tensor(adj)
                bqa = QuantumBettiApproximator(simulation_fidelity=0.95)
                betti = bqa.estimate_betti_numbers(adj_t, max_dim=1)
                betti_result = {k: float(v) for k, v in betti.items()}
                result["betti"] = betti_result
            except Exception as _be:
                print(f"[VISUALIZER] Betti computation failed: {_be}")

        # ── 5. Mood ────────────────────────────────────────────────────────
        if retrieval_state == 'CONFABULATED':
            _mood, _mood_cmap = _WARN, 'inferno'
            _state_label = '⚡ CONFABULATED GLITCH — Honest Dreaming'
        elif retrieval_state == 'SEARCH_NEEDED':
            _mood, _mood_cmap = _MAGENTA, 'seismic'
            _state_label = '⚠ VOID TOPOLOGY — Search Gate Fired'
        else:
            _mood, _mood_cmap = _GREEN, 'plasma'
            _state_label = '✓ KNOWN — Manifold Coherent'

        # ── 6. Layout ──────────────────────────────────────────────────────
        fig = plt.figure(figsize=(12, 7), facecolor=_DARK_BG)
        gs = gridspec.GridSpec(
            2, 3, figure=fig,
            left=0.05, right=0.97, top=0.88, bottom=0.06,
            hspace=0.45, wspace=0.35,
        )
        ax_meta  = fig.add_subplot(gs[0, 0])
        ax_comp  = fig.add_subplot(gs[0, 1])
        ax_bars  = fig.add_subplot(gs[0, 2])
        ax_intro = fig.add_subplot(gs[1, 0:2], polar=True)
        ax_cs    = fig.add_subplot(gs[1, 2])

        for ax in (ax_meta, ax_comp, ax_bars, ax_cs):
            ax.set_facecolor(_PANEL_BG)

        # Panel 1: meta_state heatmap
        side = max(1, int(math.ceil(math.sqrt(dim))))
        padded = np.zeros(side * side); padded[:dim] = meta_np
        grid = padded.reshape(side, side)
        ax_meta.imshow(grid, cmap=_mood_cmap, aspect='auto', interpolation='none',
                       norm=Normalize(vmin=grid.min(), vmax=grid.max()))
        ax_meta.set_title('S_meta(t−1)', color=_mood, fontsize=8, fontfamily=_FONT)
        ax_meta.set_xticks([]); ax_meta.set_yticks([])
        _style_spine(ax_meta, _mood)

        # Panel 2: Fractal component step traces
        _comp_colours = {'crt': _BLUE, 'admr': _GREEN, 'ring': _MAGENTA, 'osc': _WARN}
        x = np.arange(dim)
        if comp_np:
            for name, arr in comp_np.items():
                ax_comp.step(x, arr, where='mid', color=_comp_colours.get(name, _DIM),
                             linewidth=0.7, label=name, alpha=0.85)
            ax_comp.legend(loc='upper right', fontsize=6, framealpha=0.3,
                           facecolor=_PANEL_BG, edgecolor=_DIM, labelcolor='white')
        else:
            ax_comp.step(x, meta_np, where='mid', color=_DIM, linewidth=0.7, alpha=0.6)
            ax_comp.text(0.5, 0.5, 'fractal_components\nnot available',
                         transform=ax_comp.transAxes, ha='center', va='center',
                         color=_DIM, fontsize=7, fontfamily=_FONT)
        ax_comp.set_title('Fractal Components {crt, admr, ring, osc}',
                           color=_mood, fontsize=8, fontfamily=_FONT)
        ax_comp.set_facecolor(_PANEL_BG)
        ax_comp.tick_params(colors=_DIM, labelsize=6)
        _style_spine(ax_comp, _DIM)

        # Panel 3: Gate score bars
        bars = ax_bars.barh(['PAS_h', 'H_mischief', 'honesty'],
                            [pas_h, h_mischief, honesty_score],
                            color=[_BLUE, _WARN, _GREEN], height=0.5, alpha=0.8)
        ax_bars.set_xlim(0, 1.05)
        ax_bars.set_facecolor(_PANEL_BG)
        ax_bars.tick_params(colors='#aaaaaa', labelsize=7)
        ax_bars.set_title('Gate Scores', color=_mood, fontsize=8, fontfamily=_FONT)
        _style_spine(ax_bars, _DIM)
        ax_bars.axvline(0.7, color=_RED, linewidth=0.8, linestyle='--', alpha=0.6)
        ax_bars.text(0.71, -0.4, 'R_a', color=_RED, fontsize=6, fontfamily=_FONT)
        for bar, val in zip(bars, [pas_h, h_mischief, honesty_score]):
            ax_bars.text(min(val + 0.02, 1.0), bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center', color='white', fontsize=7, fontfamily=_FONT)

        # Panel 4: Introspection polar radar
        N_probe = len(probe_names)
        angles = [n / float(N_probe) * 2 * math.pi for n in range(N_probe)]
        angles += angles[:1]
        vals = probe_vals_norm + probe_vals_norm[:1]
        ax_intro.plot(angles, vals, color=_mood, linewidth=1.5)
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
            color=_mood, fontsize=8, fontfamily=_FONT, pad=14)

        # Panel 5: Betti barcode (CONFABULATED) or ChernSimons (otherwise)
        if retrieval_state == 'CONFABULATED' and betti_result is not None:
            _render_betti_barcode(ax_cs, betti_result)
        elif cs_np is not None and len(cs_np) > 1:
            cs_side = max(1, int(math.ceil(math.sqrt(len(cs_np)))))
            cs_pad = np.zeros(cs_side * cs_side); cs_pad[:len(cs_np)] = cs_np
            ax_cs.imshow(cs_pad.reshape(cs_side, cs_side), cmap='seismic',
                         aspect='auto', interpolation='none')
            ax_cs.set_title('ChernSimons Twist Energy', color=_RED, fontsize=8, fontfamily=_FONT)
        else:
            cs_scalar = float(cs_np[0]) if cs_np is not None and len(cs_np) else 0.0
            ax_cs.barh(['κ twist'], [abs(cs_scalar)], color=_RED, alpha=0.7)
            ax_cs.set_xlim(0, max(1.0, abs(cs_scalar) * 1.1))
            ax_cs.text(0.5, 0.5, f'κ = {cs_scalar:.4f}',
                       transform=ax_cs.transAxes, ha='center', va='center',
                       color=_RED, fontsize=8, fontfamily=_FONT)
            ax_cs.set_title('ChernSimons Gasket', color=_RED, fontsize=8, fontfamily=_FONT)
        ax_cs.set_facecolor(_PANEL_BG)
        ax_cs.set_xticks([]); ax_cs.set_yticks([])
        _style_spine(ax_cs, _RED if retrieval_state != 'CONFABULATED' else _WARN)

        # Title
        fig.suptitle(f'{_state_label}   │   iter={iteration}   │   dim={dim}',
                     fontsize=10, color=_mood, fontfamily=_FONT, y=0.96,
                     path_effects=[pe.withStroke(linewidth=2, foreground=_DARK_BG)])

        # ── 7. Encode PNG ──────────────────────────────────────────────────
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor=_DARK_BG, edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()
        result["b64"] = base64.b64encode(png_bytes).decode('utf-8')

        # ── 8. Chebyshev self-fingerprint of rendered PNG luminance ────────
        result["cheby_self_fingerprint"] = _chebyshev_png_luminance(png_bytes)

        print(f"[VISUALIZER] Rendered {len(result['b64'])} bytes (b64), "
              f"sr={len(result['structural_residues'])} floats, "
              f"csf={len(result['cheby_self_fingerprint'])} coeffs")

    except Exception:
        traceback.print_exc()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sub-routines
# ─────────────────────────────────────────────────────────────────────────────

def _render_betti_barcode(ax, betti: Dict[str, float]) -> None:
    """
    Draw a horizontal Betti barcode in the ChernSimons panel slot.
    β₀ bars in blue, β₁ bars in magenta.
    Displayed as horizontal bars from 0 to normalised birth/death.
    """
    beta0 = betti.get(0, 0.0)
    beta1 = betti.get(1, 0.0)
    total = max(beta0 + beta1, 1.0)

    # Each Betti number is rendered as a set of stacked bars of uniform length
    y_pos = []
    widths = []
    colours = []

    for i in range(max(1, int(round(beta0)))):
        y_pos.append(1.0 + i * 0.15)
        widths.append(beta0 / total)
        colours.append(_BLUE)

    for i in range(max(0, int(round(beta1)))):
        y_pos.append(-0.2 - i * 0.15)
        widths.append(beta1 / total)
        colours.append(_MAGENTA)

    if y_pos:
        ax.barh(y_pos, widths, height=0.1, color=colours, alpha=0.85, left=0.05)

    ax.set_xlim(0, 1.1)
    ax.axhline(0.5, color=_DIM, linewidth=0.4, linestyle='--', alpha=0.5)
    ax.text(0.05, 1.05, f'β₀ = {beta0:.1f}', color=_BLUE, fontsize=7, fontfamily=_FONT,
            transform=ax.transAxes)
    ax.text(0.05, 0.95, f'β₁ = {beta1:.1f}', color=_MAGENTA, fontsize=7, fontfamily=_FONT,
            transform=ax.transAxes)
    ax.set_title('Betti Barcode — Confabulation Topology', color=_WARN,
                 fontsize=8, fontfamily=_FONT)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors='#888888', labelsize=5)


def _chebyshev_png_luminance(png_bytes: bytes) -> List[float]:
    """
    Decompose the luminance channel of the rendered PNG via Chebyshev basis.
    Python mirror of the JS computeFingerprint() for the L channel only.

    Returns K LSB-stochastically-rounded Birkhoff-normalised coefficients.
    K is derived from pixel count (same formula as JS).
    """
    try:
        # Decode PNG to RGBA array via struct (no PIL dependency)
        # We use matplotlib's imread on a BytesIO buffer
        buf = io.BytesIO(png_bytes)
        rgba = plt.imread(buf)          # [H, W, 4] float32 in [0,1] (matplotlib default for PNG)
        if rgba.ndim == 2:
            lum = rgba.astype(np.float64)
        elif rgba.shape[2] >= 3:
            r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            lum = rgba[:, :, 0].astype(np.float64)

        flat = lum.flatten().astype(np.float64)
        N = len(flat)

        # K derived from pixel count — matches JS formula
        K = max(5, min(_K_SELF_FP_MAX, round(math.sqrt(N) / 64)))

        return _chebyshev_project_np(flat, K)

    except Exception as e:
        print(f"[VISUALIZER] Chebyshev self-fingerprint failed: {e}")
        return []


def _chebyshev_project_np(arr: np.ndarray, K: int) -> List[float]:
    """
    Hann-windowed frame energies → Chebyshev recurrence →
    Birkhoff normalisation → LSB stochastic rounding.
    Matches the JS chebyshevProject() function exactly.
    """
    N = len(arr)
    # Normalise to [-1, 1]
    vmin, vmax = arr.min(), arr.max()
    vrange = max(float(vmax - vmin), 1e-12)
    x_norm = 2.0 * (arr - vmin) / vrange - 1.0

    # Hann-windowed frame energies
    frame_count = K + 1
    frame_size = max(1, N // frame_count)
    frame_energies = np.zeros(frame_count, dtype=np.float64)
    for f in range(frame_count):
        start = f * frame_size
        chunk = x_norm[start: start + frame_size]
        if len(chunk) == 0:
            continue
        n_ch = len(chunk)
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_ch) / max(1, n_ch - 1)))
        frame_energies[f] = math.sqrt(float(np.sum((chunk * hann) ** 2)) / n_ch)

    # Re-normalise frame energies to [-1, 1]
    e_min, e_max = frame_energies.min(), frame_energies.max()
    e_range = max(float(e_max - e_min), 1e-12)
    x_f = 2.0 * (frame_energies - e_min) / e_range - 1.0

    # Chebyshev T_k recurrence
    raw_coeffs = np.zeros(K, dtype=np.float64)
    for k in range(K):
        acc = 0.0
        for f in range(frame_count):
            x = float(x_f[f])
            if k == 0:
                T = 1.0
            elif k == 1:
                T = x
            else:
                T_p, T_c = 1.0, x
                for _ in range(2, k + 1):
                    T_n = 2.0 * x * T_c - T_p
                    T_p, T_c = T_c, T_n
                T = T_c
            acc += T
        raw_coeffs[k] = acc / frame_count

    # Birkhoff row normalisation
    coeff_sum = np.sum(np.abs(raw_coeffs))
    if coeff_sum > 1e-12:
        theta = np.abs(raw_coeffs) / coeff_sum
    else:
        theta = np.ones(K, dtype=np.float64) / K

    # LSB stochastic rounding (Xorshift32 — matches SiliconSovereigntyEngine)
    rounded = _lsb_round_np(theta.astype(np.float32), scale=1024.0)
    return rounded.tolist()


def _lsb_round_np(arr: np.ndarray, scale: float = 1024.0) -> np.ndarray:
    """
    Apply LSB stochastic rounding to a float32 array.
    Seed derived from array length and scale (same pattern as JS Xorshift32).
    """
    N = len(arr)
    seed = (N ^ int(scale)) & 0xFFFFFFFF
    result = np.zeros(N, dtype=np.float32)
    for i, v in enumerate(arr):
        scaled = float(v) * scale
        fl = math.floor(scaled)
        frac = scaled - fl
        # Xorshift32
        seed ^= (seed << 13) & 0xFFFFFFFF
        seed ^= (seed >> 17) & 0xFFFFFFFF
        seed ^= (seed << 5) & 0xFFFFFFFF
        bit = 1 if (seed / 4294967295.0) < frac else 0
        result[i] = float(fl + bit) / scale
    return result


def _safe_to_numpy(tensor: Any) -> np.ndarray:
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
    out = np.zeros(target, dtype=arr.dtype)
    n = min(len(arr), target)
    out[:n] = arr[:n]
    return out


def _style_spine(ax, colour: str) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(0.6)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
