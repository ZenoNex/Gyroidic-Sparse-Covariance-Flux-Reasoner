import os
import pickle
import hashlib
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# -----------------------------------------------------------------------
# Blake2 digest helpers
# These mirror the exact call-sites used across the project:
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

def _blake2_summary(hex_str: str) -> str:
    """Return a human-readable annotation for a known blake2 hex string."""
    n = len(hex_str)
    if n == 20:
        return f"blake2s (digest_size=10, ingest-ID style)"
    if n == 64:
        return f"blake2s (full digest, soliton-smith identity)"
    if n == 32:
        return f"blake2b (digest_size=16, pressure-signature style)"
    return f"blake2 hex ({n} chars, variant unknown)"


def _is_blake2_hex(val) -> bool:
    if not isinstance(val, str):
        return False
    val = val.strip()
    if len(val) not in (20, 32, 64):
        return False
    try:
        int(val, 16)
        return True
    except ValueError:
        return False


# -----------------------------------------------------------------------
# PT payload schema detectors
# Three known schemas in data/encodings/:
#
#  1. soliton_smith  (DyadFossilizer.export_agent_smith)
#     type == "soliton_smith", has "blake2s_digest" at top level
#
#  2. knowledge_dyad fossil  (DyadFossilizer.fossilize)
#     type == "knowledge_dyad", keys: residue_vector, betti_0, betti_1, ...
#
#  3. interaction encoding  (EncodingManager.save_encoding)
#     keys: iteration, text_input, memory_state, response, metrics blob
# -----------------------------------------------------------------------

def _detect_pt_schema(data: dict) -> str:
    t = data.get("type", "")
    if t == "soliton_smith":
        return "soliton_smith"
    if t == "knowledge_dyad":
        return "knowledge_dyad"
    if "iteration" in data and "text_input" in data:
        return "interaction_encoding"
    return "unknown"


def _format_soliton_smith(data: dict) -> str:
    lines = ["=== SOLITON SMITH PAYLOAD ===", ""]

    b2 = data.get("blake2s_digest", "")
    if b2:
        lines.append(f"  blake2s_digest : {b2}")
        lines.append(f"  [digest type]  : {_blake2_summary(b2)}")
        lines.append("")

    pot = data.get("pot_identity_crt")
    if pot is not None:
        lines.append(f"  pot_identity_crt (Meliponini): {pot}")

    for k in ("description", "chiral_shift", "chiral_torsion", "glyphlock",
              "pestov_ionin_growth_h_gamma", "perceptual_baseline_trfc",
              "hardware_entropy_proxy", "timestamp"):
        v = data.get(k)
        if v is not None:
            lines.append(f"  {k:35s}: {v}")

    lines.append("")
    lines.append("--- Tensors ---")
    for k in ("polylog_signature", "shape_of_absence", "hyperbolic_residue",
              "gauge_field", "prime_frequencies", "gyroid_residue",
              "meta_state_shielded", "image_fingerprint", "audio_harmonics"):
        v = data.get(k)
        if v is None:
            lines.append(f"  {k:35s}: None")
        elif isinstance(v, torch.Tensor):
            lines.append(f"  {k:35s}: Tensor {list(v.shape)} dtype={v.dtype}")
        else:
            lines.append(f"  {k:35s}: {type(v).__name__}")

    b = data.get("betti_signature_8")
    if b is not None:
        lines.append(f"  betti_signature_8               : {b}")

    video = data.get("video_breather")
    if video is not None:
        lines.append(f"  video_breather                  : {video}")

    all_shapes = data.get("all_shapes")
    if all_shapes:
        lines.append(f"  all_shapes                      : {len(all_shapes)} shape tensors")

    arch = data.get("archetype_profile")
    if arch:
        lines.append(f"  archetype_profile               : {arch}")

    for k in ("agent_smith_iters", "agent_smith_gauge"):
        v = data.get(k)
        if v is not None:
            lines.append(f"  {k:35s}: {v}")

    return "\n".join(lines)


def _format_knowledge_dyad(data: dict) -> str:
    lines = ["=== KNOWLEDGE DYAD FOSSIL ===", ""]

    for k in ("text_input", "description", "timestamp"):
        v = data.get(k)
        if v:
            lines.append(f"  {k:35s}: {v}")

    lines.append("")
    lines.append("--- Topological Invariants ---")
    for k in ("chiral_score", "chiral_torsion", "glyphlock",
              "spectral_pressure", "spectral_entropy",
              "twist_energy", "seam_tension",
              "atrophy_detected", "seed_state_variance"):
        v = data.get(k)
        if v is not None:
            lines.append(f"  {k:35s}: {v}")

    b0 = data.get("betti_0")
    b1 = data.get("betti_1")
    if b0 is not None:
        lines.append(f"  betti_0 (8-threshold vec)       : {b0.tolist() if isinstance(b0, torch.Tensor) else b0}")
    if b1 is not None:
        lines.append(f"  betti_1 (8-threshold vec)       : {b1.tolist() if isinstance(b1, torch.Tensor) else b1}")

    lines.append("")
    lines.append("--- Tensors ---")
    for k in ("meta_state", "residue_vector", "hyperbolic_residue",
              "gyroid_residue", "unified_spectral_signature",
              "image_fingerprint", "audio_harmonics"):
        v = data.get(k)
        if v is None:
            lines.append(f"  {k:35s}: None")
        elif isinstance(v, torch.Tensor):
            lines.append(f"  {k:35s}: Tensor {list(v.shape)} dtype={v.dtype}")
        else:
            lines.append(f"  {k:35s}: {type(v).__name__}")

    video = data.get("video_breather")
    if video is not None:
        lines.append(f"  video_breather                  : {video}")

    metrics = data.get("metrics")
    if metrics:
        lines.append("")
        lines.append("--- Metrics ---")
        for mk, mv in metrics.items():
            lines.append(f"  {mk:35s}: {mv}")

    meta = data.get("dyad_metadata")
    if meta:
        lines.append("")
        lines.append(f"  dyad_metadata: {meta}")

    return "\n".join(lines)


def _format_interaction_encoding(data: dict) -> str:
    lines = ["=== INTERACTION ENCODING ===", ""]
    for k in ("iteration", "timestamp", "text_input", "response"):
        v = data.get(k)
        if v is not None:
            snippet = str(v)[:120]
            lines.append(f"  {k:35s}: {snippet}")

    lines.append("")
    lines.append("--- Tensors ---")
    for k in ("input_tensor", "memory_state", "final_seed_state",
              "unified_spectral_signature"):
        v = data.get(k)
        if v is None:
            continue
        elif isinstance(v, torch.Tensor):
            lines.append(f"  {k:35s}: Tensor {list(v.shape)} dtype={v.dtype}")
        else:
            lines.append(f"  {k:35s}: {type(v).__name__}")

    lines.append("")
    lines.append("--- All Remaining Keys ---")
    shown = {"iteration", "timestamp", "text_input", "response",
             "input_tensor", "memory_state", "final_seed_state",
             "unified_spectral_signature"}
    for k, v in data.items():
        if k in shown:
            continue
        if isinstance(v, torch.Tensor):
            lines.append(f"  {k:35s}: Tensor {list(v.shape)}")
        elif isinstance(v, str) and _is_blake2_hex(v):
            lines.append(f"  {k:35s}: {v}  [{_blake2_summary(v)}]")
        else:
            lines.append(f"  {k:35s}: {str(v)[:120]}")

    return "\n".join(lines)


def _format_generic_dict(data: dict, filename: str) -> str:
    lines = [f"=== DICT ({filename}) ===", f"Keys: {len(data)}", ""]
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            lines.append(f"  {str(k):35s}: Tensor {list(v.shape)} dtype={v.dtype}")
        elif isinstance(v, str) and _is_blake2_hex(v):
            lines.append(f"  {str(k):35s}: {v}  [{_blake2_summary(v)}]")
        elif isinstance(v, dict):
            lines.append(f"  {str(k):35s}: dict({len(v)} keys)")
        elif isinstance(v, (list, tuple)):
            lines.append(f"  {str(k):35s}: {type(v).__name__}[{len(v)}]")
        else:
            snippet = str(v)[:120]
            lines.append(f"  {str(k):35s}: {snippet}")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Main viewer
# -----------------------------------------------------------------------

class DataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Gyroidic Sparse Covariance Flux Reasoner - Data Inspector")
        self.root.geometry("1100x750")

        self.base_path = r"D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\data\encodings"
        if not os.path.exists(self.base_path):
            self.base_path = os.getcwd()

        self.setup_ui()
        self.refresh_file_list()

    def setup_ui(self):
        # Top toolbar
        self.toolbar = ttk.Frame(self.root, padding="5")
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.toolbar, text="Select Directory", command=self.select_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Refresh", command=self.refresh_file_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Verify Blake2", command=self.verify_blake2).pack(side=tk.LEFT, padx=5)

        self.path_label = ttk.Label(self.toolbar, text=f"Path: {self.base_path}", font=('Helvetica', 9))
        self.path_label.pack(side=tk.LEFT, padx=10)

        # Schema badge label
        self.schema_label = ttk.Label(self.toolbar, text="", font=('Helvetica', 9, 'bold'), foreground="#2266aa")
        self.schema_label.pack(side=tk.RIGHT, padx=10)

        # PanedWindow
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left Panel: File List
        self.left_frame = ttk.Frame(self.paned, padding="10")
        self.paned.add(self.left_frame, width=320)

        ttk.Label(self.left_frame, text="Available Files:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)

        list_frame = ttk.Frame(self.left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(list_frame, font=('Consolas', 9))
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=v_scroll.set)

        h_scroll = ttk.Scrollbar(self.left_frame, orient=tk.HORIZONTAL, command=self.file_listbox.xview)
        h_scroll.pack(fill=tk.X)
        self.file_listbox.config(xscrollcommand=h_scroll.set)

        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Right Panel: Data Content
        self.right_frame = ttk.Frame(self.paned, padding="10")
        self.paned.add(self.right_frame)

        ttk.Label(self.right_frame, text="Data Structure / Content Preview:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)

        self.text_area = tk.Text(self.right_frame, wrap=tk.NONE, font=('Consolas', 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)

        text_v_scroll = ttk.Scrollbar(self.right_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        text_v_scroll.place(in_=self.text_area, relx=1.0, relheight=1.0, anchor=tk.NE)
        self.text_area.config(yscrollcommand=text_v_scroll.set)

        text_h_scroll = ttk.Scrollbar(self.right_frame, orient=tk.HORIZONTAL, command=self.text_area.xview)
        text_h_scroll.pack(fill=tk.X)
        self.text_area.config(xscrollcommand=text_h_scroll.set)

        # Tag for blake2 highlight
        self.text_area.tag_configure("blake2", foreground="#aa5500", font=('Consolas', 10, 'bold'))

    def select_directory(self):
        new_path = filedialog.askdirectory(initialdir=self.base_path)
        if new_path:
            self.base_path = new_path
            self.path_label.config(text=f"Path: {self.base_path}")
            self.refresh_file_list()

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        if not os.path.exists(self.base_path):
            return
        try:
            files = sorted(os.listdir(self.base_path))
            for f in files:
                self.file_listbox.insert(tk.END, f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not list directory:\n{e}")

    def copy_to_clipboard(self):
        try:
            content = self.text_area.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("Success", "Content copied to clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy to clipboard:\n{e}")

    def verify_blake2(self):
        """
        Reads the selected .pt file and, if it contains a blake2s_digest
        (soliton_smith schema), recomputes the expected digest from the
        stored fields and compares it. Useful for integrity checking.
        """
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Select a .pt file first.")
            return

        filename = self.file_listbox.get(selection[0])
        filepath = os.path.join(self.base_path, filename)

        if not filename.endswith('.pt'):
            messagebox.showinfo("Not a PT file", "Blake2 verification only applies to .pt files.")
            return

        try:
            data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        if not isinstance(data, dict):
            messagebox.showinfo("Not a dict", "File payload is not a dict; cannot verify.")
            return

        stored = data.get("blake2s_digest")
        if not stored:
            messagebox.showinfo("No digest", "This file does not contain a blake2s_digest key.\n"
                                "Only soliton_smith payloads carry a blake2s_digest.")
            return

        # Recompute: digest_str = f"{timestamp}_{description}_{betti_numbers}_{pot_id}"
        ts = data.get("timestamp", "")
        desc = data.get("description", "")
        betti = data.get("betti_signature_8", {})
        pot = data.get("pot_identity_crt", ())
        digest_str = f"{ts}_{desc}_{betti}_{pot}"
        expected = hashlib.blake2s(digest_str.encode('utf-8')).hexdigest()

        if expected == stored:
            messagebox.showinfo("Digest VALID",
                                f"Blake2s digest verified.\n\nStored : {stored}\nExpected: {expected}")
        else:
            messagebox.showwarning("Digest MISMATCH",
                                   f"Blake2s digest does NOT match!\n\n"
                                   f"Stored : {stored}\nExpected: {expected}\n\n"
                                   f"The fossil may have been modified or was created with "
                                   f"different field values.")

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return

        filename = self.file_listbox.get(selection[0])
        filepath = os.path.join(self.base_path, filename)

        self.text_area.delete(1.0, tk.END)
        self.schema_label.config(text="")

        try:
            file_size = os.path.getsize(filepath)
        except Exception:
            file_size = 0

        # Size guard for non-binary files
        if file_size > 10 * 1024 * 1024 and not filename.endswith(('.pt', '.pkl')):
            self.text_area.insert(tk.END, f"File too large to preview ({file_size / 1024 / 1024:.2f} MB)")
            return

        try:
            if filename.endswith('.pt'):
                data = torch.load(filepath, map_location='cpu', weights_only=False)
                self._display_pt(data, filename)
            elif filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self._display_generic(data, filename)
            else:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(10000)
                    self.text_area.insert(tk.END,
                                          f"--- Raw Text Preview ({filename}) ---\n\n{content}")
                    if file_size > 10000:
                        self.text_area.insert(tk.END, "\n\n... [TRUNCATED] ...")
                except Exception as e:
                    self.text_area.insert(tk.END,
                                          f"Binary file or unreadable as text: {filename}\nError: {e}")
        except Exception as e:
            self.text_area.insert(tk.END, f"Error reading file: {e}")

    # ------------------------------------------------------------------
    # PT rendering
    # ------------------------------------------------------------------

    def _display_pt(self, data, filename):
        self.text_area.insert(tk.END, f"--- PT File: {filename} ---\n")
        self.text_area.insert(tk.END, f"    Size on disk: {os.path.getsize(os.path.join(self.base_path, filename)) / 1024:.1f} KB\n\n")

        if isinstance(data, torch.Tensor):
            info = (f"Type: PyTorch Tensor\n"
                    f"Shape: {data.shape}\n"
                    f"Device: {data.device}\n"
                    f"Dtype: {data.dtype}\n\n"
                    f"Content Preview:\n{data}")
            self.text_area.insert(tk.END, info)
            self.schema_label.config(text="[Raw Tensor]")
            return

        if not isinstance(data, dict):
            info = f"Type: {type(data)}\n\nContent:\n{data}"
            self.text_area.insert(tk.END, info)
            return

        # Detect schema
        schema = _detect_pt_schema(data)
        self.schema_label.config(text=f"Schema: {schema}")

        if schema == "soliton_smith":
            rendered = _format_soliton_smith(data)
        elif schema == "knowledge_dyad":
            rendered = _format_knowledge_dyad(data)
        elif schema == "interaction_encoding":
            rendered = _format_interaction_encoding(data)
        else:
            rendered = _format_generic_dict(data, filename)

        self.text_area.insert(tk.END, rendered)

        # Highlight any blake2 hex strings that appear in the rendered text
        self._highlight_blake2_in_text()

    def _display_generic(self, data, filename):
        self.text_area.insert(tk.END, f"--- Object Inspector ({filename}) ---\n\n")
        if isinstance(data, torch.Tensor):
            info = (f"Type: PyTorch Tensor\nShape: {data.shape}\n"
                    f"Device: {data.device}\nDtype: {data.dtype}\n\nContent Preview:\n{data}")
        elif isinstance(data, dict):
            info = _format_generic_dict(data, filename)
        elif isinstance(data, list):
            info = f"Type: List\nLength: {len(data)}\n\nPreview (First 5 items):\n{data[:5]}"
        else:
            info = f"Type: {type(data)}\n\nContent:\n{data}"
        self.text_area.insert(tk.END, info)
        self._highlight_blake2_in_text()

    def _highlight_blake2_in_text(self):
        """Walk the text widget content and tag any known-length hex strings."""
        content = self.text_area.get(1.0, tk.END)
        import re
        # Match 20, 32, or 64 lowercase hex chars that look like blake2 digests
        pattern = re.compile(r'\b([0-9a-f]{64}|[0-9a-f]{32}|[0-9a-f]{20})\b')
        for m in pattern.finditer(content):
            start_idx = f"1.0 + {m.start()} chars"
            end_idx   = f"1.0 + {m.end()} chars"
            self.text_area.tag_add("blake2", start_idx, end_idx)


if __name__ == "__main__":
    root = tk.Tk()
    app = DataViewer(root)
    root.mainloop()
