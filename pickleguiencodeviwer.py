import sys
import os
sys.path.insert(0, os.getcwd())

import re
import pickle
import hashlib
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import math

# Import codebase classes for parsing/type checks
try:
    from src.core.zeitgeist_router import ZeitgeistState, ZeitgeistRouter
    from src.core.veto_subspace import VetoSubspace, VetoLevel, RecoveryStatus, VetoSignal, VetoResult
    from src.core.love_vector import LoveVector
    from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer
    _HAS_IMPORTS = True
except Exception as e:
    _HAS_IMPORTS = False
    print(f"[IMPORTS] Failed to load codebase classes: {e}")

# Matplotlib setup
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

# Palette
_DARK_BG  = '#0a0a0a'
_PANEL_BG = '#0f1219'
_BLUE     = '#00f2ff'
_GREEN    = '#00ff41'
_MAGENTA  = '#ff00f2'
_WARN     = '#ffcc00'
_RED      = '#ff3131'
_DIM      = '#444444'
_FONT     = 'monospace'

def _style_spine(ax, colour: str) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(0.6)
    ax.tick_params(colors='#aaaaaa', labelsize=6)

# -----------------------------------------------------------------------
# Blake2 digest helpers
# Call-sites found in the project:
#
#  DETERMINISM_AND_PERSISTENCE.md  (ingest IDs):
#    hashlib.blake2s(payload, digest_size=10).hexdigest()  -> 20-char hex
#
#  knowledge_dyad_fossilizer.py  export_agent_smith():
#    hashlib.blake2s(digest_str.encode()).hexdigest()       -> 64-char hex
#    stored as payload["blake2s_digest"]
#
#  diegetic_backend.py  pressure-signature:
#    hashlib.blake2b(joined, digest_size=16).hexdigest()   -> 32-char hex
# -----------------------------------------------------------------------

_BLAKE2_RE = re.compile(r'\b([0-9a-f]{64}|[0-9a-f]{32}|[0-9a-f]{20})\b')

def _blake2_annotation(hex_str: str) -> str:
    n = len(hex_str)
    if n == 20:  return "blake2s digest_size=10  [ingest-ID style]"
    if n == 64:  return "blake2s full digest     [soliton-smith identity]"
    if n == 32:  return "blake2b digest_size=16  [pressure-signature style]"
    return f"blake2? ({n} chars)"

def _is_blake2_hex(val) -> bool:
    if not isinstance(val, str): return False
    s = val.strip()
    if len(s) not in (20, 32, 64): return False
    try: int(s, 16); return True
    except ValueError: return False

def _sanitize_latex(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Map common LaTeX formatting commands that are unsupported in mathtext to math equivalents
    t = text.replace(r'\texttt', r'\mathtt').replace(r'\textbf', r'\mathbf').replace(r'\textit', r'\mathit')
    try:
        from matplotlib.mathtext import MathTextParser
        parser = MathTextParser('path')
        parser.parse(t)
        return t
    except Exception:
        # If parsing still fails (e.g. invalid symbols, unclosed braces, unsupported LaTeX),
        # escape all dollar signs to force matplotlib to treat it as plain text.
        return text.replace('$', '\\$')


# -----------------------------------------------------------------------
# Value renderer - NO truncation. Full tensor values always shown.
#
# Doc-sourced field semantics:
#  image_fingerprint        : [96] float  - L/Cr/Cb Chebyshev (K_IMAGE_MAX=32, 3ch)
#  audio_harmonics          : [K] float   - K_AUDIO_MAX=64 Chebyshev harmonics
#  unified_spectral_signature: [96] float - merged spectral fingerprint
#  betti_0 / betti_1        : [8] float   - IHC 8-threshold filtration vecs
#  residue_vector           : [1,dim]     - chirality-redistributed seed_state
#  hyperbolic_residue       : [1,dim]     - Poincare disk projection of residue
#  gyroid_residue           : [n,n]       - irreducible entanglement matrix
#  meta_state               : [1,dim]     - architecture state at fossilization
#  pot_identity_crt         : (r61,r67,r71) - Meliponini prime residue tuple
#  polylog_signature        : from compute_polylog_signature(prime_frequencies)
#  shape_of_absence         : from compute_vacuum_residue(gyroid_residue)
# -----------------------------------------------------------------------

def _render_value(v, indent: str = "    ", label: str = "") -> str:
    """Return a full, untruncated string representation of any value."""
    prefix = f"{indent}{label + ': ' if label else ''}"

    if v is None:
        return prefix + "None"

    # Codebase classes custom rendering
    if _HAS_IMPORTS:
        if isinstance(v, ZeitgeistState):
            lines = [prefix + f"ZeitgeistState (CRT index: {v.crt_index})"]
            lines.append(indent + f"  mode        : {v.mode}")
            lines.append(indent + f"  step        : {v.step}")
            lines.append(indent + f"  level       : {v.level}")
            lines.append(indent + f"  moduli      : {v.moduli}")
            lines.append(indent + f"  alpha (diag): {v.alpha}")
            lines.append(indent + f"  braid_word  : {v.braid_word}")
            lines.append(indent + f"  cs_phase    : {v.cs_phase:.6f}")
            lines.append(_render_value(v.alpha_tensor, indent=indent + "  ", label="alpha_tensor"))
            return "\n".join(lines)

        if isinstance(v, VetoResult):
            lines = [prefix + f"VetoResult (status: {v.status.value})"]
            lines.append(indent + f"  active_vetoes     : {v.active_vetoes}")
            lines.append(indent + f"  recovery_attempted: {v.recovery_attempted}")
            lines.append(indent + f"  recovery_succeeded: {v.recovery_succeeded}")
            lines.append(indent + f"  final_severity    : {v.final_severity:.4f}")
            lines.append(_render_value(v.signals, indent=indent + "  ", label="signals"))
            lines.append(_render_value(v.budget_gates, indent=indent + "  ", label="budget_gates"))
            return "\n".join(lines)

        if isinstance(v, VetoSignal):
            return (f"{prefix}VetoSignal(source={v.source}, level={v.level.value}, "
                    f"severity={v.severity:.4f}, triggered={v.triggered}, "
                    f"can_recover={v.can_recover}, metadata={v.metadata})")

        if isinstance(v, LoveVector):
            return f"{prefix}LoveVector(dim={v.dim}, L_norm={v.L.norm().item():.6f})"

        if isinstance(v, KnowledgeDyad):
            lines = [prefix + f"KnowledgeDyad (description: '{v.linguistic_description}')"]
            lines.append(indent + f"  relevance_score: {v.relevance_score}")
            lines.append(indent + f"  timestamp      : {v.timestamp}")
            if v.image_fingerprint is not None:
                lines.append(_render_value(v.image_fingerprint, indent=indent + "  ", label="image_fingerprint"))
            if v.audio_harmonics is not None:
                lines.append(_render_value(v.audio_harmonics, indent=indent + "  ", label="audio_harmonics"))
            if v.unified_spectral_signature is not None:
                lines.append(_render_value(v.unified_spectral_signature, indent=indent + "  ", label="unified_spectral_signature"))
            if v.gyroid_residue is not None:
                lines.append(_render_value(v.gyroid_residue, indent=indent + "  ", label="gyroid_residue"))
            if v.hyperbolic_residue is not None:
                lines.append(_render_value(v.hyperbolic_residue, indent=indent + "  ", label="hyperbolic_residue"))
            if v.meta_state is not None:
                lines.append(_render_value(v.meta_state, indent=indent + "  ", label="meta_state"))
            if v.all_shapes is not None:
                lines.append(_render_value(v.all_shapes, indent=indent + "  ", label="all_shapes"))
            if v.metadata is not None:
                lines.append(_render_value(v.metadata, indent=indent + "  ", label="metadata"))
            return "\n".join(lines)

    if isinstance(v, torch.Tensor):
        hdr = f"Tensor  shape={list(v.shape)}  dtype={v.dtype}  device={v.device}"
        elems = v.numel()
        if elems == 0:
            return prefix + hdr + "\n" + indent + "  (empty tensor)"
        flat = v.detach().cpu().float().flatten()
        stats = (f"  min={flat.min().item():.6g}  max={flat.max().item():.6g}"
                 f"  mean={flat.mean().item():.6g}  std={flat.std().item():.6g}")
        # Full values - no truncation. Use high threshold so PyTorch prints all.
        with torch.no_grad():
            import io, contextlib
            buf = io.StringIO()
            torch.set_printoptions(threshold=10**9, linewidth=120, sci_mode=False)
            with contextlib.redirect_stdout(buf):
                print(v.detach().cpu())
            torch.set_printoptions(profile='default')
            vals = buf.getvalue().rstrip()
        return f"{prefix}{hdr}\n{indent}  {stats}\n{vals}"

    if isinstance(v, dict):
        if not v:
            return prefix + "dict  (empty)"
        lines = [prefix + f"dict  ({len(v)} keys)"]
        for dk, dv in v.items():
            lines.append(_render_value(dv, indent=indent + "  ", label=str(dk)))
        return "\n".join(lines)

    if isinstance(v, (list, tuple)):
        tp = type(v).__name__
        if not v:
            return prefix + f"{tp}  (empty)"
        lines = [prefix + f"{tp}  ({len(v)} items)"]
        for i, item in enumerate(v):
            lines.append(_render_value(item, indent=indent + "  ", label=f"[{i}]"))
        return "\n".join(lines)

    if isinstance(v, str):
        if _is_blake2_hex(v):
            return prefix + f'"{v}"\n{indent}  ^-- {_blake2_annotation(v)}'
        return prefix + repr(v)

    if isinstance(v, float):
        return prefix + f"{v:.8g}"

    return prefix + str(v)

# -----------------------------------------------------------------------
# Schema detection
# -----------------------------------------------------------------------

def _detect_schema(data: dict) -> str:
    t = data.get("type", "")
    if t == "soliton_smith":       return "soliton_smith"
    if t == "knowledge_dyad":      return "knowledge_dyad"
    if "iteration" in data and "text_input" in data:
        return "interaction_encoding"
    return "generic_dict"

# -----------------------------------------------------------------------
# Per-schema formatters  structured header + full values below
# -----------------------------------------------------------------------

def _fmt_section(title: str) -> str:
    bar = "=" * (len(title) + 4)
    return f"\n{bar}\n  {title}\n{bar}\n"

def _format_soliton_smith(data: dict) -> str:
    out = [_fmt_section("SOLITON SMITH  (DyadFossilizer.export_agent_smith)")]
    b2 = data.get("blake2s_digest", "")
    if b2:
        out.append(f"  blake2s_digest : {b2}")
        out.append(f"                   ^-- {_blake2_annotation(b2)}")
        out.append(f"  Digest formula : blake2s( f\"{{timestamp}}_{{description}}_{{betti_signature_8}}_{{pot_identity_crt}}\" )")
        out.append("")

    scalar_keys = [
        "pot_identity_crt", "description", "timestamp",
        "chiral_shift", "chiral_torsion", "glyphlock",
        "pestov_ionin_growth_h_gamma", "perceptual_baseline_trfc",
        "hardware_entropy_proxy", "agent_smith_iters", "agent_smith_gauge",
        "betti_signature_8", "archetype_profile", "video_breather",
    ]
    out.append("--- Scalar / Metadata ---")
    for k in scalar_keys:
        v = data.get(k)
        if v is not None:
            out.append(_render_value(v, indent="  ", label=k))
    out.append("")

    tensor_keys = [
        "polylog_signature", "shape_of_absence", "hyperbolic_residue",
        "gauge_field", "prime_frequencies", "gyroid_residue",
        "meta_state_shielded", "image_fingerprint", "audio_harmonics",
    ]
    out.append("--- Tensors (full values) ---")
    for k in tensor_keys:
        v = data.get(k)
        out.append(_render_value(v, indent="  ", label=k))

    all_shapes = data.get("all_shapes")
    if all_shapes:
        out.append(f"\n  all_shapes: list of {len(all_shapes)} tensors")
        for i, t in enumerate(all_shapes):
            out.append(_render_value(t, indent="    ", label=f"[{i}]"))

    out.append("")
    out.append("--- All remaining keys ---")
    shown = set(scalar_keys + tensor_keys + ["blake2s_digest", "all_shapes", "type",
                                              "dyad_metadata", "warmstart_states"])
    for k, v in data.items():
        if k not in shown:
            out.append(_render_value(v, indent="  ", label=k))

    meta = data.get("dyad_metadata")
    if meta is not None:
        out.append(_render_value(meta, indent="  ", label="dyad_metadata"))

    return "\n".join(out)

def _format_knowledge_dyad(data: dict) -> str:
    out = [_fmt_section("KNOWLEDGE DYAD FOSSIL  (DyadFossilizer.fossilize)")]
    out.append("--- Identity ---")
    for k in ("text_input", "description", "timestamp", "type"):
        v = data.get(k)
        if v is not None:
            out.append(_render_value(v, indent="  ", label=k))

    out.append("\n--- Topological Invariants ---")
    inv_keys = [
        "chiral_score", "chiral_torsion", "glyphlock",
        "spectral_pressure", "spectral_entropy", "twist_energy", "seam_tension",
        "atrophy_detected", "seed_state_variance",
    ]
    for k in inv_keys:
        v = data.get(k)
        if v is not None:
            out.append(_render_value(v, indent="  ", label=k))

    for k in ("betti_0", "betti_1"):
        v = data.get(k)
        if v is not None:
            out.append(_render_value(v, indent="  ", label=f"{k} (8-threshold filtration vec)"))

    out.append("\n--- Tensors (full values) ---")
    tensor_keys = [
        "meta_state", "residue_vector", "hyperbolic_residue",
        "gyroid_residue", "unified_spectral_signature",
        "image_fingerprint", "audio_harmonics",
    ]
    for k in tensor_keys:
        v = data.get(k)
        out.append(_render_value(v, indent="  ", label=k))

    v = data.get("video_breather")
    if v is not None:
        out.append(_render_value(v, indent="  ", label="video_breather"))

    metrics = data.get("metrics")
    if metrics:
        out.append("\n--- Metrics ---")
        out.append(_render_value(metrics, indent="  ", label="metrics"))

    meta = data.get("dyad_metadata")
    if meta is not None:
        out.append(_render_value(meta, indent="  ", label="dyad_metadata"))

    return "\n".join(out)

def _format_interaction_encoding(data: dict) -> str:
    out = [_fmt_section("INTERACTION ENCODING  (EncodingManager.save_encoding)")]
    out.append("--- Header ---")
    for k in ("iteration", "timestamp", "text_input", "response", "commutativity"):
        v = data.get(k)
        if v is not None:
            out.append(_render_value(v, indent="  ", label=k))

    out.append("\n--- Tensors (full values) ---")
    tensor_keys = ("input_tensor", "memory_state", "final_seed_state",
                   "unified_spectral_signature")
    shown = set(("iteration", "timestamp", "text_input", "response",
                 "commutativity") + tensor_keys)
    for k in tensor_keys:
        v = data.get(k)
        if v is not None:
            out.append(_render_value(v, indent="  ", label=k))

    out.append("\n--- All remaining keys ---")
    for k, v in data.items():
        if k not in shown:
            out.append(_render_value(v, indent="  ", label=k))

    return "\n".join(out)

def _format_generic_dict(data: dict, filename: str) -> str:
    out = [_fmt_section(f"DICT  {filename}  ({len(data)} keys)")]
    for k, v in data.items():
        out.append(_render_value(v, indent="  ", label=str(k)))
    return "\n".join(out)

def _dyad_to_dict(dyad) -> dict:
    return {
        "type": "knowledge_dyad",
        "description": getattr(dyad, "linguistic_description", ""),
        "text_input": getattr(dyad, "linguistic_description", ""),
        "image_fingerprint": getattr(dyad, "image_fingerprint", None),
        "audio_harmonics": getattr(dyad, "audio_harmonics", None),
        "video_breather": getattr(dyad, "video_breather", None),
        "unified_spectral_signature": getattr(dyad, "unified_spectral_signature", None),
        "gyroid_residue": getattr(dyad, "gyroid_residue", None),
        "hyperbolic_residue": getattr(dyad, "hyperbolic_residue", None),
        "meta_state": getattr(dyad, "meta_state", None),
        "all_shapes": getattr(dyad, "all_shapes", None),
        "relevance_score": getattr(dyad, "relevance_score", 1.0),
        "timestamp": getattr(dyad, "timestamp", ""),
        "dyad_metadata": getattr(dyad, "metadata", None),
        "chiral_score": getattr(dyad, "chiral_score", 0.0),
        "chiral_torsion": getattr(dyad, "chiral_torsion", 0.0),
        "glyphlock": getattr(dyad, "glyphlock", False),
        "spectral_pressure": getattr(dyad, "spectral_pressure", 0.0),
        "spectral_entropy": getattr(dyad, "spectral_entropy", 0.0),
        "twist_energy": getattr(dyad, "twist_energy", 0.0),
        "seam_tension": getattr(dyad, "seam_tension", 0.0),
        "betti_0": getattr(dyad, "betti_0", None),
        "betti_1": getattr(dyad, "betti_1", None),
        "atrophy_detected": getattr(dyad, "atrophy_detected", False)
    }

# -----------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------

class DataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Gyroidic Flux Reasoner - Data Inspector")
        self.root.geometry("1150x780")

        self.base_path = r"D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\data\encodings"
        if not os.path.exists(self.base_path):
            self.base_path = os.getcwd()

        self.viz_canvas_widget = None

        self.setup_ui()
        self.refresh_file_list()

    # ------------------------------------------------------------------
    def setup_ui(self):
        self.toolbar = ttk.Frame(self.root, padding="5")
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.toolbar, text="Select Directory",  command=self.select_directory).pack(side=tk.LEFT, padx=4)
        ttk.Button(self.toolbar, text="Refresh",           command=self.refresh_file_list).pack(side=tk.LEFT, padx=4)
        ttk.Button(self.toolbar, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=4)
        ttk.Button(self.toolbar, text="Verify Blake2",     command=self.verify_blake2).pack(side=tk.LEFT, padx=4)

        # Sort Order dropdown
        ttk.Label(self.toolbar, text="Sort:").pack(side=tk.LEFT, padx=(12, 2))
        self.sort_var = tk.StringVar(value="Date Modified (Newest)")
        self.sort_combo = ttk.Combobox(
            self.toolbar,
            textvariable=self.sort_var,
            values=["Date Modified (Newest)", "Date Modified (Oldest)", "Name (A-Z)", "Name (Z-A)"],
            state="readonly",
            width=22
        )
        self.sort_combo.pack(side=tk.LEFT, padx=4)
        self.sort_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_file_list())

        self.path_label = ttk.Label(self.toolbar, text=f"Path: {self.base_path}", font=('Helvetica', 9))
        self.path_label.pack(side=tk.LEFT, padx=8)

        self.schema_label = ttk.Label(self.toolbar, text="", font=('Helvetica', 9, 'bold'), foreground="#1155aa")
        self.schema_label.pack(side=tk.RIGHT, padx=8)

        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left: file list
        lf = ttk.Frame(self.paned, padding="8")
        self.paned.add(lf, width=310)
        ttk.Label(lf, text="Files:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)

        lbox_frame = ttk.Frame(lf)
        lbox_frame.pack(fill=tk.BOTH, expand=True)
        self.file_listbox = tk.Listbox(lbox_frame, font=('Consolas', 9))
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(lbox_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=vsb.set)
        hsb = ttk.Scrollbar(lf, orient=tk.HORIZONTAL, command=self.file_listbox.xview)
        hsb.pack(fill=tk.X)
        self.file_listbox.config(xscrollcommand=hsb.set)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Right: Tabbed Notebook
        rf = ttk.Frame(self.paned, padding="8")
        self.paned.add(rf)
        
        self.notebook = ttk.Notebook(rf)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Text Inspector
        self.text_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.text_tab, text=" Text Inspector ")
        
        self.text_area = tk.Text(self.text_tab, wrap=tk.NONE, font=('Consolas', 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        tv = ttk.Scrollbar(self.text_tab, orient=tk.VERTICAL, command=self.text_area.yview)
        tv.place(in_=self.text_area, relx=1.0, relheight=1.0, anchor=tk.NE)
        self.text_area.config(yscrollcommand=tv.set)
        th = ttk.Scrollbar(self.text_tab, orient=tk.HORIZONTAL, command=self.text_area.xview)
        th.pack(fill=tk.X)
        self.text_area.config(xscrollcommand=th.set)

        # Tab 2: Topological Visualizer
        self.visualizer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visualizer_tab, text=" Manifold Visualizer ")
        
        self.placeholder_label = ttk.Label(
            self.visualizer_tab,
            text="Select a PyTorch (.pt) file or state snapshot to generate manifold visualizations.",
            font=('Consolas', 10),
            foreground="#666666"
        )
        self.placeholder_label.pack(expand=True)

        # Text tags
        self.text_area.tag_configure("blake2",  foreground="#aa4400", font=('Consolas', 10, 'bold'))
        self.text_area.tag_configure("section", foreground="#004488", font=('Consolas', 10, 'bold'))
        self.text_area.tag_configure("key",     foreground="#006622", font=('Consolas', 10))

    # ------------------------------------------------------------------
    def select_directory(self):
        p = filedialog.askdirectory(initialdir=self.base_path)
        if p:
            self.base_path = p
            self.path_label.config(text=f"Path: {self.base_path}")
            self.refresh_file_list()

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        if not os.path.exists(self.base_path):
            return
        try:
            files_with_time = []
            for f in os.listdir(self.base_path):
                fpath = os.path.join(self.base_path, f)
                if os.path.isfile(fpath):
                    try:
                        mtime = os.path.getmtime(fpath)
                    except Exception:
                        mtime = 0
                    files_with_time.append((f, mtime))
            
            sort_val = self.sort_var.get()
            if sort_val == "Date Modified (Newest)":
                files_with_time.sort(key=lambda x: x[1], reverse=True)
            elif sort_val == "Date Modified (Oldest)":
                files_with_time.sort(key=lambda x: x[1], reverse=False)
            elif sort_val == "Name (A-Z)":
                files_with_time.sort(key=lambda x: x[0].lower(), reverse=False)
            elif sort_val == "Name (Z-A)":
                files_with_time.sort(key=lambda x: x[0].lower(), reverse=True)
            
            for f, _ in files_with_time:
                self.file_listbox.insert(tk.END, f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not list directory:\n{e}")

    def copy_to_clipboard(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.text_area.get(1.0, tk.END))
            messagebox.showinfo("Copied", "Content copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    def verify_blake2(self):
        sel = self.file_listbox.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Select a .pt file first.")
            return
        fname = self.file_listbox.get(sel[0])
        if not fname.endswith('.pt'):
            messagebox.showinfo("N/A", "Blake2 verification applies only to .pt files.")
            return
        fpath = os.path.join(self.base_path, fname)
        try:
            data = torch.load(fpath, map_location='cpu', weights_only=False)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        if not isinstance(data, dict):
            messagebox.showinfo("N/A", "Payload is not a dict.")
            return
        stored = data.get("blake2s_digest")
        if not stored:
            messagebox.showinfo(
                "No digest",
                "No blake2s_digest key found.\n"
                "Only soliton_smith payloads carry this field."
            )
            return
        ts   = data.get("timestamp", "")
        desc = data.get("description", "")
        betti = data.get("betti_signature_8", {})
        pot   = data.get("pot_identity_crt", ())
        digest_str = f"{ts}_{desc}_{betti}_{pot}"
        expected = hashlib.blake2s(digest_str.encode('utf-8')).hexdigest()
        if expected == stored:
            messagebox.showinfo("VALID", f"Digest matches.\n\n{stored}")
        else:
            messagebox.showwarning(
                "MISMATCH",
                f"Digest does NOT match!\n\n"
                f"Stored  : {stored}\n"
                f"Expected: {expected}"
            )

    # ------------------------------------------------------------------
    def on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        fname = self.file_listbox.get(sel[0])
        fpath = os.path.join(self.base_path, fname)

        self.text_area.delete(1.0, tk.END)
        self.schema_label.config(text="")

        try:
            fsize = os.path.getsize(fpath)
        except Exception:
            fsize = 0

        if fsize > 10 * 1024 * 1024 and not fname.endswith(('.pt', '.pkl')):
            self.text_area.insert(tk.END, f"File too large ({fsize/1024/1024:.1f} MB) - select .pt/.pkl for binary files.")
            return

        try:
            if fname.endswith('.pt'):
                data = torch.load(fpath, map_location='cpu', weights_only=False)
                self._show_pt(data, fname, fsize)
                self.update_visualization(data, fname)
            elif fname.endswith('.pkl'):
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                self._show_object(data, fname)
                self.update_visualization(data, fname)
            else:
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(10000)
                    self.text_area.insert(tk.END, f"--- Text: {fname} ---\n\n{content}")
                    if fsize > 10000:
                        self.text_area.insert(tk.END, "\n\n... [TRUNCATED] ...")
                except Exception as e:
                    self.text_area.insert(tk.END, f"Binary/unreadable: {fname}\n{e}")
                self.clear_visualization()
        except Exception as e:
            self.text_area.insert(tk.END, f"Error loading file:\n{e}")
            self.clear_visualization()

    # ------------------------------------------------------------------
    def _show_pt(self, data, fname, fsize):
        self.text_area.insert(tk.END, f"File : {fname}\nSize : {fsize/1024:.1f} KB\n")

        if isinstance(data, torch.Tensor):
            self.schema_label.config(text="[Raw Tensor]")
            self.text_area.insert(tk.END, _render_value(data, indent="", label="tensor"))
            return

        if not isinstance(data, dict):
            self.schema_label.config(text=f"[{type(data).__name__}]")
            self.text_area.insert(tk.END, _render_value(data, indent=""))
            return

        schema = _detect_schema(data)
        
        # Check if actually gyroid_state.pt
        if schema == "generic_dict" and {'iteration', 'hidden_state', 'love_invariant', 'cavity_M'}.issubset(data.keys()):
            self.schema_label.config(text="Schema: gyroid_state")
        else:
            self.schema_label.config(text=f"Schema: {schema}")

        if schema == "soliton_smith":
            rendered = _format_soliton_smith(data)
        elif schema == "knowledge_dyad":
            rendered = _format_knowledge_dyad(data)
        elif schema == "interaction_encoding":
            rendered = _format_interaction_encoding(data)
        else:
            rendered = _format_generic_dict(data, fname)

        self.text_area.insert(tk.END, rendered)
        self._apply_tags()

    def _show_object(self, data, fname):
        self.text_area.insert(tk.END, f"--- {fname} ---\n\n")
        if isinstance(data, dict):
            self.text_area.insert(tk.END, _format_generic_dict(data, fname))
        elif isinstance(data, torch.Tensor):
            self.text_area.insert(tk.END, _render_value(data))
        elif isinstance(data, list):
            self.text_area.insert(tk.END, _render_value(data))
        else:
            self.text_area.insert(tk.END, _render_value(data, indent=""))
        self._apply_tags()

    def _apply_tags(self):
        content = self.text_area.get(1.0, tk.END)
        for m in _BLAKE2_RE.finditer(content):
            s = f"1.0 + {m.start()} chars"
            e = f"1.0 + {m.end()} chars"
            self.text_area.tag_add("blake2", s, e)
        for m in re.finditer(r'^=+.*$', content, re.MULTILINE):
            s = f"1.0 + {m.start()} chars"
            e = f"1.0 + {m.end()} chars"
            self.text_area.tag_add("section", s, e)

    # ------------------------------------------------------------------
    # Visualization Logic
    # ------------------------------------------------------------------
    def clear_visualization(self):
        if self.viz_canvas_widget:
            self.viz_canvas_widget.pack_forget()
            self.viz_canvas_widget.destroy()
            self.viz_canvas_widget = None
        self.placeholder_label.config(
            text="Select a PyTorch (.pt) file or state snapshot to generate manifold visualizations.",
            foreground="#666666"
        )
        self.placeholder_label.pack(expand=True)

    def update_visualization(self, data, fname):
        if self.viz_canvas_widget:
            self.viz_canvas_widget.pack_forget()
            self.viz_canvas_widget.destroy()
            self.viz_canvas_widget = None
        self.placeholder_label.pack_forget()

        try:
            fig = plt.Figure(figsize=(8, 6), facecolor=_DARK_BG)
            
            # Re-map KnowledgeDyad class instance to dictionary schema
            if _HAS_IMPORTS and isinstance(data, KnowledgeDyad):
                data = _dyad_to_dict(data)

            is_gyroid_state = False
            if isinstance(data, dict):
                required_keys = {'iteration', 'hidden_state', 'love_invariant', 'cavity_M'}
                if required_keys.issubset(data.keys()):
                    is_gyroid_state = True
            
            if is_gyroid_state:
                self.draw_gyroid_state_dashboard(fig, data)
            elif _HAS_IMPORTS and isinstance(data, ZeitgeistState):
                self.draw_zeitgeist_state_dashboard(fig, data, fname)
            elif _HAS_IMPORTS and isinstance(data, VetoResult):
                self.draw_veto_result_dashboard(fig, data, fname)
            elif _HAS_IMPORTS and isinstance(data, LoveVector):
                self.draw_love_vector_dashboard(fig, data, fname)
            elif isinstance(data, dict) and (_detect_schema(data) in ("soliton_smith", "knowledge_dyad", "interaction_encoding")):
                self.draw_fossil_encoding_dashboard(fig, data, fname)
            elif isinstance(data, torch.Tensor):
                self.draw_raw_tensor(fig, data, fname)
            elif isinstance(data, dict):
                self.draw_generic_dict(fig, data, fname)
            else:
                raise ValueError("Data structure is not supported for high-fidelity visualization.")
            
            canvas = FigureCanvasTkAgg(fig, master=self.visualizer_tab)
            self.viz_canvas_widget = canvas.get_tk_widget()
            self.viz_canvas_widget.pack(fill=tk.BOTH, expand=True)
            canvas.draw()
        except Exception as e:
            self.placeholder_label.config(
                text=f"Manifold Visualizer failed to render this object type:\n\n{e}",
                foreground=_RED
            )
            self.placeholder_label.pack(expand=True)

    def draw_gyroid_state_dashboard(self, fig, data):
        gs = gridspec.GridSpec(3, 2, figure=fig, left=0.08, right=0.95, top=0.90, bottom=0.08, hspace=0.45, wspace=0.3)
        
        # Subplot 1: [K=8, residue_dim=32] channel view of hidden_state & hidden_state_scarred
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(_PANEL_BG)
        h = data.get("hidden_state")
        h_scarred = data.get("hidden_state_scarred")
        
        if h is not None:
            # Reshape flat state_dim back into its [K, residue_dim] view to reveal independent channels
            h_np = h.detach().cpu().float().numpy().reshape(8, 32)
            if h_scarred is not None:
                h_sc_np = h_scarred.detach().cpu().float().numpy().reshape(8, 32)
                combined = np.vstack((h_np, h_sc_np))
                ax1.imshow(combined, cmap='inferno', aspect='auto', interpolation='none')
                ax1.set_title("Channels [K=8, R=32] (hidden top, scarred bot)", color=_BLUE, fontsize=8, fontfamily=_FONT)
            else:
                ax1.imshow(h_np, cmap='inferno', aspect='auto', interpolation='none')
                ax1.set_title("Channels [K=8, residue_dim=32]", color=_BLUE, fontsize=8, fontfamily=_FONT)
        else:
            ax1.text(0.5, 0.5, "No hidden_state", color=_DIM, ha='center', va='center')
        ax1.set_xticks([]); ax1.set_yticks([])
        _style_spine(ax1, _BLUE)

        # Subplot 2: love_invariant polar subspace
        ax2 = fig.add_subplot(gs[0, 1], polar=True)
        ax2.set_facecolor(_PANEL_BG)
        love = data.get("love_invariant")
        if love is not None:
            love_np = love.detach().cpu().float().numpy().flatten()
            n_love = len(love_np)
            angles = np.linspace(0, 2*np.pi, n_love, endpoint=False)
            ax2.plot(angles, love_np, color=_GREEN, linewidth=1)
            ax2.fill(angles, love_np, color=_GREEN, alpha=0.15)
            ax2.set_title("love_invariant (64-dim Subspace)", color=_GREEN, fontsize=8, fontfamily=_FONT, pad=10)
        else:
            ax2.text(0.5, 0.5, "No love_invariant", color=_DIM, ha='center', va='center')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.spines['polar'].set_color(_DIM)
        ax2.grid(color=_DIM, linewidth=0.4, alpha=0.4)

        # Subplot 3: Resonance Cavity modes M & D_dark
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(_PANEL_BG)
        cav_M = data.get("cavity_M")
        cav_D = data.get("cavity_D_dark")
        if cav_M is not None:
            m_modes = cav_M.detach().cpu().float().mean(dim=-1).numpy()
            x_modes = np.arange(m_modes.shape[1] if m_modes.ndim > 1 else len(m_modes))
            colors = [_BLUE, _GREEN, _MAGENTA, _WARN, _RED]
            for i in range(min(5, m_modes.shape[0])):
                ax3.step(x_modes, m_modes[i], where='mid', color=colors[i], linewidth=0.8, alpha=0.8, label=f"F_{i}")
            
            if cav_D is not None:
                d_modes = cav_D.detach().cpu().float().mean(dim=-1).numpy()
                ax3.step(x_modes, d_modes.mean(axis=0) if d_modes.ndim > 1 else d_modes, where='mid', color='white', linewidth=1.2, linestyle='--', alpha=0.7, label="D_dark")
                
            ax3.set_title("Resonance Cavity Spectral Modes (mean)", color=_MAGENTA, fontsize=8, fontfamily=_FONT)
            ax3.legend(loc='upper right', fontsize=5, framealpha=0.3, facecolor=_PANEL_BG, labelcolor='white')
        else:
            ax3.text(0.5, 0.5, "No cavity modes", color=_DIM, ha='center', va='center')
        ax3.tick_params(colors=_DIM, labelsize=6)
        _style_spine(ax3, _DIM)

        # Subplot 4: Zeitgeist Router (Matrioshka/CRT & alpha_tensor)
        sub_gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.1)
        ax4_left = fig.add_subplot(sub_gs4[0, 0])
        ax4_right = fig.add_subplot(sub_gs4[0, 1])
        ax4_left.set_facecolor(_PANEL_BG)
        ax4_right.set_facecolor(_PANEL_BG)
        
        zeit = data.get("zeitgeist")
        if zeit is not None:
            # 1. Matrioshka & CRT Geometry (Left)
            ax4_left.set_aspect('equal')
            max_level = 5
            curr_level = getattr(zeit, 'level', 0)
            moduli = getattr(zeit, 'moduli', [1])
            num_sectors = len(moduli) if moduli else 1
            crt_idx = getattr(zeit, 'crt_index', 0)
            active_sector = crt_idx % num_sectors if num_sectors > 0 else 0
            
            theta = np.linspace(0, 2*np.pi, 100)
            for L in range(max_level + 1):
                r = 0.2 + 0.8 * (L / max_level)
                color = _MAGENTA if L == curr_level else _DIM
                linewidth = 1.0 if L == curr_level else 0.3
                alpha_val = 0.8 if L == curr_level else 0.2
                ax4_left.plot(r * np.cos(theta), r * np.sin(theta), color=color, linewidth=linewidth, alpha=alpha_val)
            
            sector_angle = 2 * np.pi / num_sectors
            start_angle = active_sector * sector_angle
            end_angle = (active_sector + 1) * sector_angle
            
            r_outer = 1.0
            theta_fill = np.linspace(start_angle, end_angle, 20)
            x_fill = np.concatenate([[0], r_outer * np.cos(theta_fill), [0]])
            y_fill = np.concatenate([[0], r_outer * np.sin(theta_fill), [0]])
            ax4_left.fill(x_fill, y_fill, color=_GREEN, alpha=0.3)
            
            for i in range(num_sectors):
                ang = i * sector_angle
                ax4_left.plot([0, r_outer * np.cos(ang)], [0, r_outer * np.sin(ang)], color=_DIM, linewidth=0.3, linestyle='--')
            
            ax4_left.set_xlim(-1.1, 1.1)
            ax4_left.set_ylim(-1.1, 1.1)
            ax4_left.set_xticks([]); ax4_left.set_yticks([])
            ax4_left.set_title(f"Matrioshka L{curr_level} / CRT {active_sector}", color=_MAGENTA, fontsize=6, fontfamily=_FONT)
            _style_spine(ax4_left, _MAGENTA)
            
            # 2. Alpha Tensor Heatmap (Right)
            if hasattr(zeit, "alpha_tensor") and zeit.alpha_tensor is not None:
                alpha_np = zeit.alpha_tensor.detach().cpu().float().numpy()
                ax4_right.imshow(alpha_np, cmap='seismic', aspect='auto', interpolation='none')
                ax4_right.set_title(f"alpha_tensor (mode={getattr(zeit, 'mode', 'N/A')})", color=_WARN, fontsize=6, fontfamily=_FONT)
            else:
                ax4_right.text(0.5, 0.5, "No alpha_tensor", color=_DIM, ha='center', va='center', fontsize=6)
            ax4_right.set_xticks([]); ax4_right.set_yticks([])
            _style_spine(ax4_right, _WARN)
        else:
            ax4_left.text(0.5, 0.5, "No Router state", color=_DIM, ha='center', va='center', fontsize=6)
            ax4_right.text(0.5, 0.5, "No Router state", color=_DIM, ha='center', va='center', fontsize=6)
            ax4_left.set_xticks([]); ax4_left.set_yticks([])
            ax4_right.set_xticks([]); ax4_right.set_yticks([])
            _style_spine(ax4_left, _DIM)
            _style_spine(ax4_right, _DIM)

        # Subplot 5: Temporal Model Trust & State-Scarred Cross-Correlation
        sub_gs5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0], wspace=0.35)
        ax5_left = fig.add_subplot(sub_gs5[0, 0])
        ax5_right = fig.add_subplot(sub_gs5[0, 1])
        ax5_left.set_facecolor(_PANEL_BG)
        ax5_right.set_facecolor(_PANEL_BG)
        
        t_state = data.get("temporal_model_state")
        trust = None
        if isinstance(t_state, dict):
            trust = t_state.get("trust_scalars")
            
        if trust is not None:
            trust_np = trust.detach().cpu().float().numpy().flatten()
            ax5_left.bar(np.arange(len(trust_np)), trust_np, color=_BLUE, alpha=0.8, width=0.4)
            ax5_left.set_ylim(0, max(1.1, float(trust_np.max()) * 1.1))
            ax5_left.set_title("Trust Scalars", color=_BLUE, fontsize=7, fontfamily=_FONT)
        else:
            dmg = data.get("damage_residue")
            if dmg is not None:
                dmg_np = dmg.detach().cpu().float().numpy().flatten()
                ax5_left.plot(dmg_np, color=_RED, linewidth=0.7)
                ax5_left.set_title("Damage Residue", color=_RED, fontsize=7, fontfamily=_FONT)
            else:
                ax5_left.text(0.5, 0.5, "No trust data", color=_DIM, ha='center', va='center', fontsize=7)
        ax5_left.tick_params(colors=_DIM, labelsize=5)
        _style_spine(ax5_left, _DIM)

        h = data.get("hidden_state")
        h_sc = data.get("hidden_state_scarred")
        if h is not None and h_sc is not None:
            h_np = h.detach().cpu().float().numpy().flatten()
            h_sc_np = h_sc.detach().cpu().float().numpy().flatten()
            h_norm = (h_np - h_np.mean()) / (h_np.std() + 1e-8)
            h_sc_norm = (h_sc_np - h_sc_np.mean()) / (h_sc_np.std() + 1e-8)
            cross = np.correlate(h_norm, h_sc_norm, mode='full') / (len(h_norm) + 1e-8)
            lags = np.arange(-len(h_sc_norm) + 1, len(h_norm))
            ax5_right.plot(lags, cross, color=_WARN, linewidth=0.7)
            ax5_right.fill_between(lags, cross, color=_WARN, alpha=0.1)
            peak_idx = np.argmax(np.abs(cross))
            peak_val = cross[peak_idx]
            peak_lag = lags[peak_idx]
            ax5_right.plot(peak_lag, peak_val, 'o', color=_RED, markersize=3)
            ax5_right.axvline(0, color='white', linestyle='--', linewidth=0.4, alpha=0.5)
            ax5_right.text(0.05, 0.9, f"Peak: {peak_val:.2f}\nLag: {peak_lag}",
                           transform=ax5_right.transAxes, color='white', fontsize=5, fontfamily=_FONT)
            ax5_right.set_title("State vs Scarred", color=_WARN, fontsize=7, fontfamily=_FONT)
        else:
            ax5_right.text(0.5, 0.5, "No state correlation", color=_DIM, ha='center', va='center', fontsize=7)
        ax5_right.tick_params(colors=_DIM, labelsize=5)
        _style_spine(ax5_right, _DIM)

        # Subplot 6: Veto Subspace & Instability Diagnostics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor(_PANEL_BG)
        
        diagnostics = {}
        if zeit is not None and hasattr(zeit, "to_dict"):
            try:
                diagnostics = zeit.to_dict().get("diagnostics", {})
            except Exception:
                pass
                
        clock_dt = diagnostics.get("clock_dt", 1.0)
        valence = diagnostics.get("valence", 0.0)
        nc_curv = diagnostics.get("nc_curvature", 0.0)
        grazing_p = diagnostics.get("grazing_pressure", 0.0)
        
        metrics = ['clock_dt', 'valence', 'nc_curv', 'grazing_p']
        vals = [float(clock_dt or 0.0), float(valence or 0.0), float(nc_curv or 0.0), float(grazing_p or 0.0)]
        
        ax6.barh(metrics, vals, color=[_GREEN, _MAGENTA, _RED, _WARN], height=0.5, alpha=0.85)
        ax6.set_title("Veto Subspace & Instability", color=_WARN, fontsize=8, fontfamily=_FONT)
        ax6.tick_params(colors=_DIM, labelsize=6)
        for i, val in enumerate(vals):
            ax6.text(val + 0.01, i, f"{val:.4f}", va='center', color='white', fontsize=6, fontfamily=_FONT)
        _style_spine(ax6, _DIM)

        fig.suptitle(f"Global gyroid_state.pt  (iteration {data.get('iteration', 'N/A')})",
                     color='white', fontsize=11, fontfamily=_FONT, y=0.97)

    def draw_fossil_encoding_dashboard(self, fig, data, fname):
        gs = gridspec.GridSpec(3, 2, figure=fig, left=0.08, right=0.95, top=0.90, bottom=0.08, hspace=0.45, wspace=0.3)
        schema = _detect_schema(data)
        
        # Subplot 1: Topological cross-correlation comparison to current gyroid_state.pt
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(_PANEL_BG)
        
        gyroid_path = os.path.join(os.path.dirname(self.base_path), "gyroid_state.pt")
        if not os.path.exists(gyroid_path):
            gyroid_path = os.path.join(os.getcwd(), "gyroid_state.pt")
            
        global_state = None
        if os.path.exists(gyroid_path):
            try:
                global_state = torch.load(gyroid_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"[VISUALIZER] Failed to load global gyroid state: {e}")
        
        cand_tensor = None
        for k in ("memory_state", "residue_vector", "meta_state", "final_seed_state"):
            if k in data and isinstance(data[k], torch.Tensor):
                cand_tensor = data[k].detach().cpu().float().flatten()
                break
                
        if cand_tensor is not None and global_state is not None and "hidden_state" in global_state:
            g_tensor = global_state["hidden_state"].detach().cpu().float().flatten().numpy()
            cand_np = cand_tensor.numpy()
            c_mean, g_mean = cand_np.mean(), g_tensor.mean()
            c_std, g_std = cand_np.std(), g_tensor.std()
            c_norm = (cand_np - c_mean) / (c_std + 1e-8)
            g_norm = (g_tensor - g_mean) / (g_std + 1e-8)
            cross_corr = np.correlate(c_norm, g_norm, mode='full') / (max(len(c_norm), len(g_norm)) + 1e-8)
            lags = np.arange(-len(g_norm) + 1, len(c_norm))
            ax1.plot(lags, cross_corr, color=_BLUE, linewidth=0.8, alpha=0.9)
            ax1.fill_between(lags, cross_corr, color=_BLUE, alpha=0.15)
            peak_idx = np.argmax(np.abs(cross_corr))
            peak_lag = lags[peak_idx]
            peak_val = cross_corr[peak_idx]
            ax1.plot(peak_lag, peak_val, 'o', color=_RED, markersize=3)
            ax1.axvline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.6)
            cos_sim = float(np.dot(cand_np, g_tensor) / (np.linalg.norm(cand_np) * np.linalg.norm(g_tensor) + 1e-8))
            ax1.text(0.05, 0.95, f"CosSim: {cos_sim:.3f}\nPeak: {peak_val:.3f}\nLag: {peak_lag}",
                     transform=ax1.transAxes, color='white', fontsize=6, fontfamily=_FONT,
                     verticalalignment='top')
            ax1.set_title("Cross-Correlation (vs. Live State)", color=_BLUE, fontsize=8, fontfamily=_FONT)
        else:
            if cand_tensor is not None:
                cand_np = cand_tensor.numpy()
                c_norm = (cand_np - cand_np.mean()) / (cand_np.std() + 1e-8)
                auto_corr = np.correlate(c_norm, c_norm, mode='full') / (len(c_norm) + 1e-8)
                lags = np.arange(-len(c_norm) + 1, len(c_norm))
                ax1.plot(lags, auto_corr, color=_BLUE, linewidth=0.8, alpha=0.9)
                ax1.fill_between(lags, auto_corr, color=_BLUE, alpha=0.15)
                ax1.axvline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.6)
                ax1.set_title("Auto-Correlation (Fossil Vector)", color=_BLUE, fontsize=8, fontfamily=_FONT)
            else:
                ax1.text(0.5, 0.5, "No comparison available", color=_DIM, ha='center', va='center')
        ax1.tick_params(colors=_DIM, labelsize=5)
        _style_spine(ax1, _BLUE)

        # Subplot 2: Topological Invariants (8 parameters)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(_PANEL_BG)
        
        inv_labels = ['chiral_s', 'chiral_t', 'glyphlock', 'spec_p', 'spec_e', 'twist_e', 'seam_t', 'atrophy']
        inv_vals = [
            float(data.get("chiral_score") or 0.0),
            float(data.get("chiral_torsion") or 0.0),
            1.0 if data.get("glyphlock") else 0.0,
            float(data.get("spectral_pressure") or 0.0),
            float(data.get("spectral_entropy") or 0.0),
            float(data.get("twist_energy") or 0.0),
            float(data.get("seam_tension") or 0.0),
            float(data.get("atrophy_detected") or 0.0)
        ]
        
        y_pos2 = np.arange(len(inv_labels))
        ax2.barh(y_pos2, inv_vals, color=_GREEN, alpha=0.8, height=0.5)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(inv_labels, color='white', fontsize=6, fontfamily=_FONT)
        ax2.set_title("Topological Invariants", color=_GREEN, fontsize=8, fontfamily=_FONT)
        ax2.tick_params(colors=_DIM, labelsize=6)
        _style_spine(ax2, _DIM)

        # Subplot 3: Betti Curves (beta_0 & beta_1 filtration waveforms)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(_PANEL_BG)
        b0 = data.get("betti_0")
        b1 = data.get("betti_1")
        if b0 is not None or b1 is not None:
            if b0 is not None:
                b0_np = b0.detach().cpu().float().numpy().flatten() if isinstance(b0, torch.Tensor) else np.array(b0)
                ax3.step(np.arange(len(b0_np)), b0_np, where='mid', color=_BLUE, linewidth=1.0, label="beta_0")
            if b1 is not None:
                b1_np = b1.detach().cpu().float().numpy().flatten() if isinstance(b1, torch.Tensor) else np.array(b1)
                ax3.step(np.arange(len(b1_np)), b1_np, where='mid', color=_MAGENTA, linewidth=1.0, label="beta_1")
            ax3.set_title("Betti IHC 8-Filtration Curves", color=_MAGENTA, fontsize=8, fontfamily=_FONT)
            ax3.legend(loc='upper right', fontsize=5, framealpha=0.3, facecolor=_PANEL_BG, labelcolor='white')
        else:
            ax3.text(0.5, 0.5, "No Betti vectors available", color=_DIM, ha='center', va='center')
        ax3.tick_params(colors=_DIM, labelsize=6)
        _style_spine(ax3, _DIM)

        # Subplot 4: Chebyshev Fingerprint split-plot (Coefficients & Cross-Correlation with Live Resonance)
        sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.5)
        ax4_top = fig.add_subplot(sub_gs[0])
        ax4_bottom = fig.add_subplot(sub_gs[1])
        ax4_top.set_facecolor(_PANEL_BG)
        ax4_bottom.set_facecolor(_PANEL_BG)
        
        img_fp = data.get("image_fingerprint")
        aud_harm = data.get("audio_harmonics")
        l_chan = data.get("L")
        cr_chan = data.get("Cr")
        cb_chan = data.get("Cb")
        aud_chan = data.get("chebyshev_harmonics")
        uss = data.get("unified_spectral_signature")
        
        profile = None
        profile_name = ""
        
        if img_fp is not None:
            profile = img_fp.detach().cpu().float().numpy().flatten()
            profile_name = "Image Chebyshev"
            ax4_top.plot(profile, color=_BLUE, linewidth=0.8)
            ax4_top.set_title(f"{profile_name} Coefficients", color=_BLUE, fontsize=7, fontfamily=_FONT)
        elif l_chan is not None:
            ax4_top.plot(l_chan, color='white', linewidth=0.8, label="L")
            if cr_chan is not None: ax4_top.plot(cr_chan, color=_RED, linewidth=0.8, label="Cr")
            if cb_chan is not None: ax4_top.plot(cb_chan, color=_BLUE, linewidth=0.8, label="Cb")
            ax4_top.legend(loc='upper right', fontsize=4, framealpha=0.3, facecolor=_PANEL_BG, labelcolor='white')
            ax4_top.set_title("Multimodal Chebyshev Channels", color=_BLUE, fontsize=7, fontfamily=_FONT)
            profile = np.array(l_chan)
            profile_name = "L Chebyshev"
        elif aud_harm is not None or aud_chan is not None:
            harm = aud_harm if aud_harm is not None else aud_chan
            profile = harm.detach().cpu().float().numpy().flatten() if isinstance(harm, torch.Tensor) else np.array(harm)
            profile_name = "Audio Chebyshev"
            ax4_top.step(np.arange(len(profile)), profile, where='mid', color=_GREEN, linewidth=0.8)
            ax4_top.set_title("Audio Chebyshev Harmonics", color=_GREEN, fontsize=7, fontfamily=_FONT)
        elif uss is not None:
            profile = uss.detach().cpu().float().numpy().flatten()
            profile_name = "Unified Spectral Signature"
            ax4_top.plot(profile, color=_MAGENTA, linewidth=0.8)
            ax4_top.set_title(f"{profile_name} Coefficients", color=_MAGENTA, fontsize=7, fontfamily=_FONT)
        else:
            ax4_top.text(0.5, 0.5, "No media profile", color=_DIM, ha='center', va='center', fontsize=7)
            
        ax4_top.tick_params(colors=_DIM, labelsize=5)
        _style_spine(ax4_top, _DIM)

        omega = None
        if global_state is not None and "temporal_model_state" in global_state:
            t_state = global_state["temporal_model_state"]
            if "spectral_corrector.omega" in t_state:
                omega = t_state["spectral_corrector.omega"].detach().cpu().float().flatten().numpy()

        if profile is not None:
            if omega is not None:
                p_norm = (profile - profile.mean()) / (profile.std() + 1e-8)
                o_norm = (omega - omega.mean()) / (omega.std() + 1e-8)
                cross = np.correlate(p_norm, o_norm, mode='full') / (max(len(p_norm), len(o_norm)) + 1e-8)
                lags = np.arange(-len(o_norm) + 1, len(p_norm))
                ax4_bottom.plot(lags, cross, color=_MAGENTA, linewidth=0.7)
                ax4_bottom.fill_between(lags, cross, color=_MAGENTA, alpha=0.1)
                p_idx = np.argmax(np.abs(cross))
                p_lag = lags[p_idx]
                p_val = cross[p_idx]
                ax4_bottom.plot(p_lag, p_val, 'o', color=_RED, markersize=3)
                ax4_bottom.axvline(0, color='white', linestyle='--', linewidth=0.4, alpha=0.5)
                ax4_bottom.text(0.05, 0.9, f"Peak: {p_val:.2f} (Lag: {p_lag})",
                                transform=ax4_bottom.transAxes, color='white', fontsize=5, fontfamily=_FONT)
                ax4_bottom.set_title("Chebyshev vs Live Resonance (omega)", color=_MAGENTA, fontsize=7, fontfamily=_FONT)
            else:
                p_norm = (profile - profile.mean()) / (profile.std() + 1e-8)
                auto = np.correlate(p_norm, p_norm, mode='full') / (len(p_norm) + 1e-8)
                lags = np.arange(-len(p_norm) + 1, len(p_norm))
                ax4_bottom.plot(lags, auto, color=_MAGENTA, linewidth=0.7)
                ax4_bottom.fill_between(lags, auto, color=_MAGENTA, alpha=0.1)
                ax4_bottom.axvline(0, color='white', linestyle='--', linewidth=0.4, alpha=0.5)
                ax4_bottom.set_title("Chebyshev Auto-Correlation", color=_MAGENTA, fontsize=7, fontfamily=_FONT)
        else:
            ax4_bottom.text(0.5, 0.5, "No signal to correlate", color=_DIM, ha='center', va='center', fontsize=7)
            
        ax4_bottom.tick_params(colors=_DIM, labelsize=5)
        _style_spine(ax4_bottom, _DIM)

        # Subplot 5: Poincaré Disk Hyperbolic Trajectory Plotting
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor(_PANEL_BG)
        
        # Get raw vector for trajectory
        traj_vec = None
        for k in ("hyperbolic_residue", "residue_vector", "meta_state", "final_seed_state"):
            if k in data and data[k] is not None:
                traj_vec = data[k].detach().cpu().float().numpy().flatten()
                break
                
        if traj_vec is not None and len(traj_vec) >= 4:
            # --- DEEP UNDERSTANDING EXTRACTION ---
            # 1. Breather Modes (Magnitude of FFT)
            # 2. Phase Alignments (Angle of FFT)
            # 3. Feature Scars (Laplacian / 2nd Derivative spatial defects)
            
            # Compute FFT to unpack the interference pattern
            complex_spectrum = np.fft.rfft(traj_vec)
            amplitudes = np.abs(complex_spectrum) # Breather Modes
            phases = np.angle(complex_spectrum)   # Phase Alignments
            
            # Drop the DC component (index 0) for better topological projection
            if len(amplitudes) > 1:
                amplitudes = amplitudes[1:]
                phases = phases[1:]
                
            # Map onto Poincaré disk: r_new = tanh(amplitude), theta = phase
            r_new = np.tanh(amplitudes)
            x_proj = r_new * np.cos(phases)
            y_proj = r_new * np.sin(phases)
            
            # Find Feature Scars via discrete Laplacian on the original spatial vector
            # scar = |v[i+1] - 2v[i] + v[i-1]|
            laplacian = np.abs(np.convolve(traj_vec, [1, -2, 1], mode='same'))
            # Find indices where the scar magnitude is exceptionally high (top 5%)
            scar_threshold = np.percentile(laplacian, 95) if len(laplacian) > 0 else 0
            
            # Draw boundary circle (Poincaré unit disk boundary)
            theta_circle = np.linspace(0, 2*np.pi, 200)
            ax5.plot(np.cos(theta_circle), np.sin(theta_circle), color=_DIM, linewidth=0.8, linestyle='--')
            
            # Plot sprawling hyperbolic trajectory path (connecting the modes)
            ax5.plot(x_proj, y_proj, color=_WARN, linewidth=0.8, alpha=0.5, label="Phase Alignment Trajectory")
            
            # Scatter the modes (Breather Modes), colored by their index
            ax5.scatter(x_proj, y_proj, c=np.arange(len(x_proj)), cmap='viridis', s=10 + 20*r_new, alpha=0.8, label="Breather Modes")
            
            # Spatial trajectory: r_s = tanh(|v_i|), theta_s = i/N * 2*pi
            spatial_r = np.tanh(np.abs(traj_vec))
            spatial_theta = np.linspace(0, 2*np.pi, len(traj_vec), endpoint=False)
            x_sp = spatial_r * np.cos(spatial_theta)
            y_sp = spatial_r * np.sin(spatial_theta)
            
            # Plot spatial Feature Scars as red 'x' marks where laplacian is high
            scar_idx = np.where(laplacian > scar_threshold)[0]
            if len(scar_idx) > 0:
                ax5.scatter(x_sp[scar_idx], y_sp[scar_idx], marker='x', color=_RED, s=30, label="Feature Scars")
            
            # Highlight start/end points of the frequency trajectory
            if len(x_proj) > 0:
                ax5.plot(x_proj[0], y_proj[0], 'o', color=_GREEN, markersize=5, label="Base Frequency")
            
            # Formatting
            ax5.set_xlim(-1.1, 1.1)
            ax5.set_ylim(-1.1, 1.1)
            ax5.set_aspect('equal')
            ax5.set_title("Poincaré Hyperbolic Trajectory (Deep Extracted)", color=_WARN, fontsize=8, fontfamily=_FONT)
            ax5.legend(loc='upper right', fontsize=4, framealpha=0.3, facecolor=_PANEL_BG, labelcolor='white')
        else:
            # Fallback if no vector
            gyr = data.get("gyroid_residue")
            if gyr is not None:
                gyr_np = gyr.detach().cpu().float().numpy()
                if gyr_np.ndim == 2:
                    ax5.imshow(gyr_np, cmap='inferno', aspect='auto', interpolation='none')
                    ax5.set_title("gyroid_residue (entanglement)", color=_WARN, fontsize=8, fontfamily=_FONT)
                else:
                    ax5.plot(gyr_np.flatten(), color=_WARN, linewidth=0.8)
                    ax5.set_title("gyroid_residue (flattened)", color=_WARN, fontsize=8, fontfamily=_FONT)
            else:
                ax5.text(0.5, 0.5, "No entanglement matrix", color=_DIM, ha='center', va='center')
                ax5.set_title("Hyperbolic Projection", color=_WARN, fontsize=8, fontfamily=_FONT)
                
        ax5.set_xticks([]); ax5.set_yticks([])
        _style_spine(ax5, _WARN)

        # Subplot 6: Identity / Linguistic Anchor & Pot Tags
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor(_PANEL_BG)
        
        text_in = data.get("text_input") or data.get("description") or ""
        digest = data.get("blake2s_digest") or ""
        pot = data.get("pot_identity_crt")
        
        wrapped_text = ""
        if text_in:
            wrapped_text = "Linguistic Anchor:\n" + "\n".join(text_in[i:i+32] for i in range(0, min(96, len(text_in)), 32))
            if len(text_in) > 96:
                wrapped_text += "..."
                
        tag_info = ""
        if pot is not None:
            tag_info = f"Cerumen Pot Tag: {pot}"
            
        digest_info = ""
        if digest:
            digest_info = f"Blake2s: {digest[:16]}..."
            
        full_info = "\n\n".join(filter(None, [wrapped_text, tag_info, digest_info]))
        if not full_info:
            full_info = "No metadata available"
            
        lines = full_info.split('\n')
        num_lines = len(lines)
        line_height = 0.08
        start_y = 0.5 + (num_lines - 1) * line_height / 2.0
        
        for idx, line in enumerate(lines):
            sanitized_line = _sanitize_latex(line)
            ax6.text(0.08, start_y - idx * line_height, sanitized_line, color='white', fontsize=7, fontfamily=_FONT, va='center')
        ax6.set_title("Identity & Context", color=_BLUE, fontsize=8, fontfamily=_FONT)
        ax6.set_xticks([]); ax6.set_yticks([])
        _style_spine(ax6, _BLUE)

        try:
            fsize = os.path.getsize(os.path.join(self.base_path, fname)) / 1024.0
            size_str = f"fsize={fsize:.1f} KB"
        except Exception:
            size_str = "fsize=N/A"

        fig.suptitle(f"{schema.replace('_', ' ').upper()}  ({size_str})",
                     color='white', fontsize=11, fontfamily=_FONT, y=0.97)

    def draw_raw_tensor(self, fig, data, fname):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(_PANEL_BG)
        
        flat = data.detach().cpu().float().numpy().flatten()
        size = flat.shape[0]
        
        if data.ndim == 2:
            ax.imshow(data.detach().cpu().float().numpy(), cmap='inferno', aspect='auto', interpolation='none')
            ax.set_title(f"Raw Tensor: {fname} (shape={list(data.shape)})", color=_BLUE, fontsize=10, fontfamily=_FONT)
        else:
            ax.plot(flat, color=_BLUE, linewidth=0.8)
            ax.set_title(f"Raw Tensor Waveform (size={size})", color=_BLUE, fontsize=10, fontfamily=_FONT)
            ax.tick_params(colors=_DIM, labelsize=8)
            _style_spine(ax, _BLUE)
        fig.suptitle(f"Tensor Object", color='white', fontsize=11, fontfamily=_FONT, y=0.97)

    def draw_generic_dict(self, fig, data, fname):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(_PANEL_BG)
        
        vals = []
        labels = []
        for k, v in data.items():
            if isinstance(v, (int, float)):
                vals.append(float(v))
                labels.append(k)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                vals.append(float(v.item()))
                labels.append(k)
                
        if vals:
            y_pos = np.arange(len(vals))
            ax.barh(y_pos, vals, color=_BLUE, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, color='white', fontsize=8, fontfamily=_FONT)
            ax.set_title(f"Generic Dictionary: {fname}", color=_BLUE, fontsize=10, fontfamily=_FONT)
            ax.tick_params(colors=_DIM, labelsize=8)
            _style_spine(ax, _BLUE)
        else:
            ax.text(0.5, 0.5, f"Dictionary with {len(data)} keys\nNo scalar values to plot.",
                    color='white', ha='center', va='center', fontsize=9, fontfamily=_FONT)
            ax.set_xticks([]); ax.set_yticks([])
            _style_spine(ax, _DIM)
        fig.suptitle(f"Generic Dictionary Object", color='white', fontsize=11, fontfamily=_FONT, y=0.97)

    def draw_zeitgeist_state_dashboard(self, fig, state, fname):
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.95, top=0.90, bottom=0.08, hspace=0.45, wspace=0.3)
        
        # Subplot 1: alpha_tensor heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(_PANEL_BG)
        alpha = state.alpha_tensor
        if alpha is not None:
            alpha_np = alpha.detach().cpu().float().numpy()
            ax1.imshow(alpha_np, cmap='seismic', aspect='auto', interpolation='none')
            ax1.set_title(f"alpha_tensor (Symmetric index, mode={state.mode})", color=_WARN, fontsize=8, fontfamily=_FONT)
        else:
            ax1.text(0.5, 0.5, "No alpha_tensor", color=_DIM, ha='center', va='center')
        ax1.set_xticks([]); ax1.set_yticks([])
        _style_spine(ax1, _WARN)

        # Subplot 2: Matrioshka Shells & CRT Facet Lock-In
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(_PANEL_BG)
        ax2.set_aspect('equal')
        
        max_level = 5
        curr_level = getattr(state, 'level', 0)
        moduli = getattr(state, 'moduli', [1])
        num_sectors = len(moduli) if moduli else 1
        crt_idx = getattr(state, 'crt_index', 0)
        active_sector = crt_idx % num_sectors if num_sectors > 0 else 0
        
        # Draw concentric Matrioshka shells
        theta = np.linspace(0, 2*np.pi, 100)
        for L in range(max_level + 1):
            r = 0.2 + 0.8 * (L / max_level)
            color = _MAGENTA if L == curr_level else _DIM
            linewidth = 1.5 if L == curr_level else 0.5
            alpha_val = 0.8 if L == curr_level else 0.3
            ax2.plot(r * np.cos(theta), r * np.sin(theta), color=color, linewidth=linewidth, alpha=alpha_val)
            if L == curr_level:
                ax2.text(r*0.7, r*0.7, f"L{L}", color=_MAGENTA, fontsize=6, fontfamily=_FONT)

        # Highlight the active CRT facet sector
        sector_angle = 2 * np.pi / num_sectors
        start_angle = active_sector * sector_angle
        end_angle = (active_sector + 1) * sector_angle
        
        r_outer = 1.0
        theta_fill = np.linspace(start_angle, end_angle, 20)
        x_fill = np.concatenate([[0], r_outer * np.cos(theta_fill), [0]])
        y_fill = np.concatenate([[0], r_outer * np.sin(theta_fill), [0]])
        ax2.fill(x_fill, y_fill, color=_GREEN, alpha=0.2)
        
        for i in range(num_sectors):
            ang = i * sector_angle
            ax2.plot([0, r_outer * np.cos(ang)], [0, r_outer * np.sin(ang)], color=_DIM, linewidth=0.5, linestyle='--')
            
        # Project alpha residues into the active sector
        alpha_diag = getattr(state, 'alpha', [])
        if len(alpha_diag) > 0:
            radii = np.linspace(0.3, 0.9, len(alpha_diag))
            angles = np.linspace(start_angle + 0.1, end_angle - 0.1, len(alpha_diag))
            x_pts = radii * np.cos(angles)
            y_pts = radii * np.sin(angles)
            
            a_np = np.array([float(x) for x in alpha_diag])
            if a_np.max() > a_np.min():
                sizes = 10 + 40 * (a_np - a_np.min()) / (a_np.max() - a_np.min() + 1e-8)
            else:
                sizes = np.full_like(a_np, 20)
                
            ax2.scatter(x_pts, y_pts, s=sizes, color=_GREEN, alpha=0.8, zorder=5)
            
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(-1.1, 1.1)
        ax2.set_title(f"Matrioshka Shell (L={curr_level}) & CRT Facet ({active_sector})", color=_GREEN, fontsize=8, fontfamily=_FONT)
        _style_spine(ax2, _GREEN)

        # Subplot 3: Braid word trajectory
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(_PANEL_BG)
        word = state.braid_word
        if word:
            ax3.step(np.arange(len(word)), word, where='mid', color=_BLUE, linewidth=1.0)
            ax3.set_title(f"Braid word generators (len={len(word)})", color=_BLUE, fontsize=8, fontfamily=_FONT)
        else:
            ax3.text(0.5, 0.5, "Braid word is empty (identity e)", color=_DIM, ha='center', va='center', fontsize=8)
            ax3.set_title("Braid word generators", color=_BLUE, fontsize=8, fontfamily=_FONT)
        ax3.tick_params(colors=_DIM, labelsize=6)
        _style_spine(ax3, _BLUE)

        # Subplot 4: Metadata details
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(_PANEL_BG)
        info = [
            f"Step counter: {state.step}",
            f"Matrioshka level: {state.level}",
            f"Chern-Simons phase: {state.cs_phase:.6f}",
            f"Mode status: {state.mode}",
            f"Is undefined: {state.is_undefined}"
        ]
        y_pos = 0.8
        for line in info:
            ax4.text(0.1, y_pos, line, color='white', fontsize=8, fontfamily=_FONT, va='center')
            y_pos -= 0.15
        ax4.set_title("State Metadata", color=_MAGENTA, fontsize=8, fontfamily=_FONT)
        ax4.set_xticks([]); ax4.set_yticks([])
        _style_spine(ax4, _MAGENTA)

        fig.suptitle(f"ZeitgeistState Dashboard | {fname}", color='white', fontsize=11, fontfamily=_FONT, y=0.97)

    def draw_veto_result_dashboard(self, fig, result, fname):
        gs = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.95, top=0.90, bottom=0.08, hspace=0.45, wspace=0.3)
        
        # Subplot 1: Signal severities
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(_PANEL_BG)
        signals = result.signals
        if signals:
            sources = [s.source for s in signals]
            severities = [s.severity for s in signals]
            colors = [_RED if s.triggered else _GREEN for s in signals]
            
            y_pos = np.arange(len(sources))
            ax1.barh(y_pos, severities, color=colors, alpha=0.8, height=0.5)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(sources, color='white', fontsize=8, fontfamily=_FONT)
            ax1.set_xlim(0, 1.1)
            ax1.set_title(f"Veto Signal Severities (max={result.final_severity:.4f})", color=_BLUE, fontsize=8, fontfamily=_FONT)
            for i, s in enumerate(signals):
                trig_str = " [VETO]" if s.triggered else ""
                ax1.text(s.severity + 0.02, i, f"{s.severity:.4f}{trig_str}", va='center', color='white', fontsize=7, fontfamily=_FONT)
        else:
            ax1.text(0.5, 0.5, "No veto signals evaluated", color=_DIM, ha='center', va='center')
        _style_spine(ax1, _BLUE)

        # Subplot 2: Status & Recovery Info
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(_PANEL_BG)
        status_color = _RED if result.status == RecoveryStatus.ESCALATED else (_WARN if result.status == RecoveryStatus.RECOVERED else _GREEN)
        info = [
            f"Lattice Status: {result.status.value}",
            f"Active veto count: {result.active_vetoes}",
            f"Recovery attempted: {result.recovery_attempted}",
            f"Recovery succeeded: {result.recovery_succeeded}",
        ]
        y_pos = 0.8
        for line in info:
            ax2.text(0.1, y_pos, line, color='white', fontsize=8, fontfamily=_FONT, va='center')
            y_pos -= 0.2
        ax2.set_title("Recovery Status", color=status_color, fontsize=8, fontfamily=_FONT)
        ax2.set_xticks([]); ax2.set_yticks([])
        _style_spine(ax2, status_color)

        # Subplot 3: Budget Gates
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(_PANEL_BG)
        gates = result.budget_gates
        if gates:
            gate_names = list(gates.keys())
            gate_vals = [1.0 if v else 0.0 for v in gates.values()]
            colors = [_RED if v else _GREEN for v in gates.values()]
            
            y_pos3 = np.arange(len(gate_names))
            ax3.barh(y_pos3, gate_vals, color=colors, alpha=0.8, height=0.4)
            ax3.set_yticks(y_pos3)
            ax3.set_yticklabels(gate_names, color='white', fontsize=8, fontfamily=_FONT)
            ax3.set_xlim(0, 1.2)
            ax3.set_title("Budget Gates (red=skipped)", color=_WARN, fontsize=8, fontfamily=_FONT)
            for i, name in enumerate(gate_names):
                status_str = "BLOCKED" if gates[name] else "OK"
                ax3.text(gate_vals[i] + 0.05, i, status_str, va='center', color='white', fontsize=8, fontfamily=_FONT)
        else:
            ax3.text(0.5, 0.5, "No budget gates evaluated", color=_DIM, ha='center', va='center')
        _style_spine(ax3, _WARN)

        fig.suptitle(f"VetoResult Dashboard | {fname}", color='white', fontsize=11, fontfamily=_FONT, y=0.97)

    def draw_love_vector_dashboard(self, fig, love_vec, fname):
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(_PANEL_BG)
        
        flat = love_vec.L.detach().cpu().float().numpy().flatten()
        ax.plot(flat, color=_MAGENTA, linewidth=0.8)
        ax.set_title(f"LoveVector L profile (dim={love_vec.dim}, norm={love_vec.L.norm().item():.6f})", color=_MAGENTA, fontsize=10, fontfamily=_FONT)
        ax.tick_params(colors=_DIM, labelsize=8)
        _style_spine(ax, _MAGENTA)
        fig.suptitle(f"LoveVector Invariant Map | {fname}", color='white', fontsize=11, fontfamily=_FONT, y=0.97)


if __name__ == "__main__":
    root = tk.Tk()
    app = DataViewer(root)
    root.mainloop()
