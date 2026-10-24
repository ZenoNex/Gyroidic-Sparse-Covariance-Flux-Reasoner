"""
diegetic_visualizer.py
======================

Deep Manifold Visualization Generator for the Gyroidic Sparse Covariance Flux Reasoner.

Architectural role
------------------
This module is the *Audience Projection* operator (: M  A) expressed
as a matplotlib rendering pipeline.  It does not invent structure  it
**exposes** the exact live tensors that caused a confabulation or ego
death event, with roughness conserved.

Recursive Self-Reference contract
----------------------------------
The system CAN touch its own representations:

    1.  ``meta_state``        the FractalMetaFunctional's persistent
                               S_meta(t-1) buffer (registered on engine).
    2.  ``fractal_components``  the four sub-tensors returned by
                               FractalMetaFunctional.forward():
                               {crt, admr, ring, osc}.
    3.  ``introspection``     GeometricSelfModelProbe unit directions:
                               moral / uncertainty / creative / metacognitive.
    4.  ``chern_simons``      Gasket twist energy from ChernSimonsGasket.

Return contract
---------------
``render_manifold_fracture()`` returns a **dict** (not a bare string):

    {
        "b64": str | None,                   # base64-encoded PNG
        "structural_residues": list[float],  # LSB-rounded probe+component norms
        "cheby_self_fingerprint": list[float], # Chebyshev of PNG luminance
        "betti_matrix": list | None,         # [beta0, beta1] if CONFABULATED
    }

The backend injects these directly into meta_state (Introspection I channel)
and reads b64 for the HTTP response.

Roughness Preservation (anti-smoothing policy)
----------------------------------------------
- No interpolation or Gaussian blur.
- Colour maps ``inferno`` and ``seismic`` maximise discontinuity contrast.
- Step-plot (not line) for component waveforms.

References
----------
DIEGETIC_ENGINE.md 7, GYROID_REASONER.md 5,
fractal_meta_functional.py, RESONANCE_INTELLIGENCE_CORE.md 11
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


# 
# Colour palette
# 
_DARK_BG  = '#0a0a0a'
_PANEL_BG = '#0f1219'
_BLUE     = '#00f2ff'
_GREEN    = '#00ff41'
_MAGENTA  = '#ff00f2'
_WARN     = '#ffcc00'
_RED      = '#ff3131'
_DIM      = '#444444'
_FONT     = 'monospace'

# 
# K for the self-Chebyshev decomposition (derived from PNG pixel count below)
# 
_K_SELF_FP_MAX = 32


# 
# Public entry point
# 

def brenier_pushforward(coords: np.ndarray, potential: np.ndarray, kappa: float = 0.15) -> np.ndarray:
    """
    Brenier push-forward gradient shift: T(x) = x + kappa * \\nabla W(x).
    Displaces coordinates along the gradient of the potential/void-friction landscape.
    """
    grad_y, grad_x = np.gradient(potential)
    displaced_x = coords[..., 0] + kappa * grad_x
    displaced_y = coords[..., 1] + kappa * grad_y
    return np.stack([displaced_x, displaced_y], axis=-1)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0, 1] to OKLab (L, a, b)."""
    mask = rgb > 0.04045
    linear_rgb = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    l = 0.4122214708 * linear_rgb[..., 0] + 0.5363325363 * linear_rgb[..., 1] + 0.0514459929 * linear_rgb[..., 2]
    m = 0.2119034982 * linear_rgb[..., 0] + 0.6806995451 * linear_rgb[..., 1] + 0.1073969566 * linear_rgb[..., 2]
    s = 0.0883024619 * linear_rgb[..., 0] + 0.2817188376 * linear_rgb[..., 1] + 0.6299787005 * linear_rgb[..., 2]
    l_ = np.cbrt(np.maximum(l, 0.0))
    m_ = np.cbrt(np.maximum(m, 0.0))
    s_ = np.cbrt(np.maximum(s, 0.0))
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, b], axis=-1)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab (L, a, b) back to sRGB [0, 1]."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3
    r_lin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    linear = np.stack([r_lin, g_lin, b_lin], axis=-1)
    linear = np.clip(linear, 0.0, 1.0)
    mask = linear > 0.0031308
    srgb = np.where(mask, 1.055 * (linear ** (1.0 / 2.4)) - 0.055, 12.92 * linear)
    return np.clip(srgb, 0.0, 1.0)


def apply_oklab_luminance_absorption(rgb: np.ndarray, void_friction: float = 0.0, curvature: float = 0.0) -> np.ndarray:
    """
    Applies OKLab perceptual luminance absorption and flattened variance in gravity/void wells.
    L' = L * exp(-alpha * (void_friction + curvature))
    a' = a * exp(-beta * (void_friction + curvature))
    b' = b * exp(-beta * (void_friction + curvature))
    """
    intensity = float(void_friction) + float(curvature)
    if intensity <= 0.01:
        return rgb
    lab = srgb_to_oklab(rgb)
    alpha = 0.35
    beta = 0.55
    lab[..., 0] *= math.exp(-alpha * intensity)
    lab[..., 1] *= math.exp(-beta * intensity)
    lab[..., 2] *= math.exp(-beta * intensity)
    return oklab_to_srgb(lab)


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
    spatial_metric: Optional[Any] = None,
    void_friction: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate a base64-encoded PNG, extract structural residues and a
    Chebyshev self-fingerprint from the result, and (on CONFABULATED)
    compute a Betti barcode.

    Returns
    -------
    dict with keys:
        b64                   : str | None   base64 PNG
        structural_residues   : list[float]  LSB-rounded norms
        cheby_self_fingerprint: list[float]  Chebyshev of PNG luminance
        betti                 : dict | None  {beta0, beta1} if CONFABULATED
    """
    result: Dict[str, Any] = {
        "b64": None,
        "structural_residues": [],
        "cheby_self_fingerprint": [],
        "betti": None,
    }

    try:
        #  1. Tensor  numpy 
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

        #  2. Structural residues (probe norms + component norms) 
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

        #  3. Introspection polar values (normalised) 
        pv_max = max(probe_vals) if max(probe_vals) > 1e-8 else 1.0
        probe_vals_norm = [v / pv_max for v in probe_vals]

        #  4. Betti computation on CONFABULATED 
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

        #  5. Mood 
        if retrieval_state == 'CONFABULATED':
            _mood, _mood_cmap = _WARN, 'inferno'
            _state_label = '[POWER] CONFABULATED GLITCH  Honest Dreaming'
        elif retrieval_state == 'SEARCH_NEEDED':
            _mood, _mood_cmap = _MAGENTA, 'seismic'
            _state_label = ' VOID TOPOLOGY  Search Gate Fired'
        else:
            _mood, _mood_cmap = _GREEN, 'plasma'
            _state_label = ' KNOWN  Manifold Coherent'

        #  6. Layout 
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

        # Land rent curvature / void friction potential field W(x)
        curvature_scale = float(void_friction)
        if spatial_metric is not None:
            sm_arr = _safe_to_numpy(spatial_metric)
            if sm_arr.size > 0:
                curvature_scale += float(max(0.0, np.mean(np.diag(sm_arr)) - 1.0))

        if curvature_scale > 0.05:
            # Brenier push-forward T(x) = x + kappa * \nabla W(x)
            y_coords, x_coords = np.mgrid[0:side, 0:side].astype(np.float64)
            coords = np.stack([x_coords, y_coords], axis=-1)
            displaced_coords = brenier_pushforward(coords, grid, kappa=0.15 * min(2.0, curvature_scale))
            shift_x = int(np.clip(np.round(np.mean(displaced_coords[..., 0] - coords[..., 0])), -2, 2))
            shift_y = int(np.clip(np.round(np.mean(displaced_coords[..., 1] - coords[..., 1])), -2, 2))
            deformed_grid = np.roll(np.roll(grid, shift=shift_x, axis=1), shift=shift_y, axis=0)
            ax_meta.imshow(deformed_grid, cmap=_mood_cmap, aspect='auto', interpolation='none',
                           norm=Normalize(vmin=deformed_grid.min(), vmax=deformed_grid.max()))
            ax_meta.set_title(f'S_meta(t-1) [Brenier Shift \u03ba={0.15*min(2.0, curvature_scale):.2f}]',
                           color=_mood, fontsize=8, fontfamily=_FONT)
        else:
            ax_meta.imshow(grid, cmap=_mood_cmap, aspect='auto', interpolation='none',
                           norm=Normalize(vmin=grid.min(), vmax=grid.max()))
            ax_meta.set_title('S_meta(t-1)', color=_mood, fontsize=8, fontfamily=_FONT)
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
            'Introspection  Geometric Self-Model Probes\n'
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
            ax_cs.barh([' twist'], [abs(cs_scalar)], color=_RED, alpha=0.7)
            ax_cs.set_xlim(0, max(1.0, abs(cs_scalar) * 1.1))
            ax_cs.text(0.5, 0.5, f' = {cs_scalar:.4f}',
                       transform=ax_cs.transAxes, ha='center', va='center',
                       color=_RED, fontsize=8, fontfamily=_FONT)
            ax_cs.set_title('ChernSimons Gasket', color=_RED, fontsize=8, fontfamily=_FONT)
        ax_cs.set_facecolor(_PANEL_BG)
        ax_cs.set_xticks([]); ax_cs.set_yticks([])
        _style_spine(ax_cs, _RED if retrieval_state != 'CONFABULATED' else _WARN)

        # Title
        fig.suptitle(f'{_state_label}      iter={iteration}      dim={dim}',
                     fontsize=10, color=_mood, fontfamily=_FONT, y=0.96,
                     path_effects=[pe.withStroke(linewidth=2, foreground=_DARK_BG)])

        #  7. Encode PNG with OKLab perceptual luminance absorption in gravity/void wells
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                    facecolor=_DARK_BG, edgecolor='none')
        plt.close(fig)
        buf.seek(0)

        if curvature_scale > 0.05:
            try:
                img_rgba = plt.imread(buf)
                rgb_absorbed = apply_oklab_luminance_absorption(
                    img_rgba[:, :, :3],
                    void_friction=void_friction,
                    curvature=curvature_scale
                )
                if img_rgba.shape[2] == 4:
                    final_rgba = np.dstack([rgb_absorbed, img_rgba[:, :, 3]])
                else:
                    final_rgba = rgb_absorbed
                buf_absorbed = io.BytesIO()
                plt.imsave(buf_absorbed, final_rgba, format='png')
                buf_absorbed.seek(0)
                png_bytes = buf_absorbed.read()
            except Exception as _ok_e:
                print(f"[VISUALIZER] OKLab filtering fallback: {_ok_e}")
                buf.seek(0)
                png_bytes = buf.read()
        else:
            png_bytes = buf.read()

        result["b64"] = base64.b64encode(png_bytes).decode('utf-8')

        #  8. Chebyshev self-fingerprint of rendered PNG luminance 
        result["cheby_self_fingerprint"] = _chebyshev_png_luminance(png_bytes)

        print(f"[VISUALIZER] Rendered {len(result['b64'])} bytes (b64), "
              f"sr={len(result['structural_residues'])} floats, "
              f"csf={len(result['cheby_self_fingerprint'])} coeffs")

    except Exception:
        traceback.print_exc()

    return result


# 
# Sub-routines
# 

def _render_betti_barcode(ax, betti: Dict[str, float]) -> None:
    """
    Draw a horizontal Betti barcode in the ChernSimons panel slot.
     bars in blue,  bars in magenta.
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
    ax.text(0.05, 1.05, f' = {beta0:.1f}', color=_BLUE, fontsize=7, fontfamily=_FONT,
            transform=ax.transAxes)
    ax.text(0.05, 0.95, f' = {beta1:.1f}', color=_MAGENTA, fontsize=7, fontfamily=_FONT,
            transform=ax.transAxes)
    ax.set_title('Betti Barcode  Confabulation Topology', color=_WARN,
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

        # K derived from pixel count  matches JS formula
        K = max(5, min(_K_SELF_FP_MAX, round(math.sqrt(N) / 64)))

        return _chebyshev_project_np(flat, K)

    except Exception as e:
        print(f"[VISUALIZER] Chebyshev self-fingerprint failed: {e}")
        return []


def _chebyshev_project_np(arr: np.ndarray, K: int, basis_type: str = "chebyshev") -> List[float]:
    """
    Hann-windowed frame energies -> Orthogonal polynomial recurrence ->
    Birkhoff normalisation -> LSB stochastic rounding.
    Matches the JS chebyshevProject() function exactly when basis_type='chebyshev',
    and evaluates canonical PolynomialBasis across Legendre, Hermite, Laguerre, or B-spline.
    """
    import torch
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

    # Orthogonal polynomial recurrence via canonical PolynomialBasis
    from src.core.polynomial_coprime import PolynomialBasis
    basis = PolynomialBasis(degree=K - 1, basis_type=basis_type, domain=(-1.0, 1.0))
    t_xf = torch.from_numpy(x_f.astype(np.float32))
    # basis.evaluate returns [frame_count, K]
    T_mat = basis.evaluate(t_xf).detach().cpu().numpy()
    raw_coeffs = T_mat.mean(axis=0).astype(np.float64)

    # Birkhoff row normalisation
    coeff_sum = np.sum(np.abs(raw_coeffs))
    if coeff_sum > 1e-12:
        theta = np.abs(raw_coeffs) / coeff_sum
    else:
        theta = np.ones(K, dtype=np.float64) / K

    # LSB stochastic rounding (Xorshift32 - matches SiliconSovereigntyEngine)
    rounded = _lsb_round_np(theta.astype(np.float32), scale=1024.0)
    return rounded.tolist()


def extract_spectral_image_fingerprint(
    image_input: Any,
    K: int = 32,
    basis_type: str = "chebyshev"
) -> Dict[str, Any]:
    """
    Decomposes an image across L, Cr, Cb channels into orthogonal polynomial spectra
    (Chebyshev, Legendre, Hermite, Laguerre, or B-spline).
    Matches JS chebyshevProject() in diegetic_terminal.html and backend ingestion.
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if isinstance(image_input, str):
        if Image is not None:
            with Image.open(image_input) as img:
                rgb = img.convert('RGB')
                arr = np.array(rgb, dtype=np.float64) / 255.0
        else:
            arr = plt.imread(image_input)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] > 3:
                arr = arr[:, :, :3]
            arr = arr.astype(np.float64)
    elif Image is not None and isinstance(image_input, Image.Image):
        rgb = image_input.convert('RGB')
        arr = np.array(rgb, dtype=np.float64) / 255.0
    else:
        arr = np.asarray(image_input, dtype=np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0

    if arr.ndim == 2:
        r = g = b = arr
        h, w = arr.shape
    elif arr.shape[-1] >= 3:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        h, w = arr.shape[:2]
    else:
        r = g = b = arr[:, :, 0]
        h, w = arr.shape[:2]

    # BT.601 color transform matching diegetic_terminal.html:3912-3914
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 0.5
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5

    l_coeffs = _chebyshev_project_np(lum.flatten(), K, basis_type=basis_type)
    cr_coeffs = _chebyshev_project_np(cr.flatten(), K, basis_type=basis_type)
    cb_coeffs = _chebyshev_project_np(cb.flatten(), K, basis_type=basis_type)

    return {
        "chebyshev_degree": K,
        "basis_type": basis_type,
        "L": l_coeffs,
        "Cr": cr_coeffs,
        "Cb": cb_coeffs,
        "px_width": w,
        "px_height": h,
    }



def _lsb_round_np(arr: np.ndarray, scale: float = 1024.0) -> np.ndarray:
    """
    Apply LSB stochastic rounding to a float32 array.
    Uses Structurally Honest Jitter (Silicon Sovereignty 45.2).
    """
    from src.core.honest_jitter import harvest_honest_jitter
    import torch
    
    N = len(arr)
    # Harvest hardware entropy for the stochastic bit mask
    jitter = harvest_honest_jitter((N,), scaled=False) # [0, 1]
    jitter_np = jitter.detach().cpu().numpy()
    
    result = np.zeros(N, dtype=np.float32)
    for i, v in enumerate(arr):
        scaled = float(v) * scale
        fl = math.floor(scaled)
        frac = scaled - fl
        
        # Stochastic bit from honest jitter
        bit = 1 if jitter_np[i] < frac else 0
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
