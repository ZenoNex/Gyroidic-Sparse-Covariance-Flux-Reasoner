# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
diagnose_fossil.py — Knowledge Dyad Fossil Inspector

Answers: "Does this fossil mean anything other than text to the AI?"

Usage:
    .venv\\Scripts\\python.exe diagnose_fossil.py [path_to_fossil.pt]
    .venv\\Scripts\\python.exe diagnose_fossil.py  (auto-picks latest fossil)
"""

import sys
import os
import torch
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_DIR = os.path.join(ROOT, "data", "encodings")

# ── Pick fossil ────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    fossil_path = sys.argv[1]
else:
    fossils = sorted(glob.glob(os.path.join(ENCODINGS_DIR, "*.pt")),
                     key=os.path.getmtime, reverse=True)
    if not fossils:
        print("[ERR] No .pt fossils found in data/encodings/")
        sys.exit(1)
    fossil_path = fossils[0]

print(f"\n{'='*64}")
print(f" FOSSIL INSPECTOR: {os.path.basename(fossil_path)}")
print(f"{'='*64}\n")

data = torch.load(fossil_path, map_location="cpu")

# ── 1. Meta ────────────────────────────────────────────────────────────────────
print(f"[1] TYPE        : {data.get('type', '?')}")
print(f"[1] TIMESTAMP   : {data.get('timestamp', '?')}")
desc = data.get('description', '')
print(f"[1] DESCRIPTION : {desc[:120]}{'...' if len(desc)>120 else ''}")
print()

# ── 2. Image Fingerprint ───────────────────────────────────────────────────────
fp = data.get('image_fingerprint')
if fp is None:
    print("[2] IMAGE FINGERPRINT : *** NOT PRESENT ***")
else:
    fp = fp.float()
    l2  = fp.norm().item()
    nz  = (fp.abs() > 1e-6).sum().item()
    print(f"[2] IMAGE FINGERPRINT : shape={list(fp.shape)}")
    print(f"    L2 norm           : {l2:.6f}  {'<-- ALL ZEROS (text-only fossil)' if l2 < 1e-4 else '<-- NON-ZERO (image embedded)'}")
    print(f"    Non-zero elements : {int(nz)} / {fp.numel()}")
    if l2 > 1e-4:
        print(f"    Min / Max         : {fp.min().item():.4f} / {fp.max().item():.4f}")
        print(f"    First 8 values    : {[round(x,4) for x in fp[:8].tolist()]}")
print()

# ── 3. Residue Vector (what the AI uses for resonance search) ──────────────────
rv = data.get('residue_vector')
if rv is None:
    print("[3] RESIDUE VECTOR : *** NOT PRESENT — fossil is invalid ***")
else:
    rv = rv.float().flatten()
    print(f"[3] RESIDUE VECTOR : shape={list(rv.shape)}")
    print(f"    L2 norm        : {rv.norm().item():.6f}")
    print(f"    Mean / Std     : {rv.mean().item():.4f} / {rv.std().item():.4f}")
    print(f"    First 8 values : {[round(x,4) for x in rv[:8].tolist()]}")
print()

# ── 4. Hyperbolic Residue (Poincaré embedding) ─────────────────────────────────
hr = data.get('hyperbolic_residue')
if hr is not None:
    hr = hr.float().flatten()
    ecc = hr.norm().item()
    print(f"[4] HYPERBOLIC RESIDUE : shape={list(hr.shape)}, eccentricity={ecc:.4f}")
    print(f"    (eccentricity < 1.0 = inside Poincaré disk, stable)")
print()

# ── 5. Metrics ─────────────────────────────────────────────────────────────────
metrics = data.get('metrics', {})
print(f"[5] METRICS : {metrics}")
print()

# ── 6. Cross-modality torsion analysis ────────────────────────────────────────
print("─── Cross-Modality Torsion Analysis ───────────────────────────────────")
if fp is not None and rv is not None and l2 > 1e-4:
    # If the image fingerprint is non-zero, the residue_vector should correlate
    # with it because ResidueFusion.forward() projects fp → 512D then computes
    # torsion = tanh(matmul(img_proj - txt_proj, torsion_matrix)).
    # We can test if zeroing out the fingerprint would change the residue.
    # Load actual fusion_layer weights from the engine state file.
    # After the fix, self.fusion_layer is a registered submodule of
    # DiegeticPhysicsEngine, so its weights appear in gyroid_state.pt
    # under the key prefix "fusion_layer.*".
    import sys
    sys.path.insert(0, ROOT)
    from src.core.knowledge_dyad_fossilizer import ResidueFusion

    fusion = ResidueFusion(feature_dim=256, fingerprint_dim=137)
    state_pt = os.path.join(ROOT, "gyroid_state.pt")
    weights_loaded = False
    if os.path.exists(state_pt):
        try:
            full_state = torch.load(state_pt, map_location="cpu")
            fl_state = {k.replace("fusion_layer.", ""): v
                        for k, v in full_state.items()
                        if k.startswith("fusion_layer.")}
            if fl_state:
                fusion.load_state_dict(fl_state)
                weights_loaded = True
                print(f"  Loaded fusion_layer weights from gyroid_state.pt ({len(fl_state)} tensors)")
            else:
                print("  gyroid_state.pt exists but has no 'fusion_layer.*' keys.")
                print("  Restart the backend once to generate them, then re-run this script.")
        except Exception as e:
            print(f"  Could not load gyroid_state.pt: {e}")
    else:
        print("  gyroid_state.pt not found — using random fusion_layer (not representative).")

    fp_norm   = fp.norm().item()
    rv_norm   = rv.norm().item()
    cos_sim   = torch.dot(fp[:min(len(fp), len(rv))].flatten(),
                          rv[:min(len(fp), len(rv))].flatten()) \
                / (fp[:min(len(fp), len(rv))].norm() * rv[:min(len(fp), len(rv))].norm() + 1e-8)
    print(f"  Image fingerprint L2  : {fp_norm:.4f}")
    print(f"  Residue vector L2     : {rv_norm:.4f}")
    print(f"  Cosine sim (fp vs rv) : {cos_sim.item():.4f}")
    print()
    print("  Interpretation:")
    print("  • ResidueFusion projects fp (137D) → 512D via a Linear layer, then")
    print("    computes torsion = tanh((img_proj - txt_proj) @ torsion_matrix).")
    print("  • If fp is NON-ZERO → the residue_vector is a function of BOTH the")
    print("    image and the text (cross-modal torsion). The fossil IS multimodal.")
    print("  • If fp is ALL-ZERO → image_proj = 0, so residue = tanh(-txt_proj @")
    print("    torsion_matrix). Text-only. Image never contributed.")
elif fp is not None and l2 < 1e-4:
    print("  *** Image fingerprint is ALL-ZERO ***")
    print("  The residue_vector was computed from text embedding alone.")
    print("  This fossil is PURELY TEXT — the image was not embedded.")
    print()
    print("  Why this happens:")
    print("  1. The ASSOCIATE command was submitted WITHOUT an armed image fingerprint, OR")
    print("  2. The image was dropped AFTER submit (fingerprint cleared before routing), OR")
    print("  3. This fossil was created before the fingerprint routing fix was applied.")
    print()
    print("  To create a TRUE multimodal fossil:")
    print("  → Drop image in the Diegetic Terminal (wait for cyan glow on input box)")
    print("  → Type your description in the armed input box")
    print("  → Press Enter — the fingerprint AND text go together")
else:
    print("  (No image fingerprint to analyze)")

print()
print("─── VERDICT ──────────────────────────────────────────────────────────")
if fp is not None and l2 > 1e-4:
    print("  ✓ MULTIMODAL FOSSIL — image Chebyshev fingerprint is embedded.")
    print("    The resonance search uses a residue built from both modalities.")
else:
    print("  ✗ TEXT-ONLY FOSSIL — residue is purely linguistic.")
    print("    Re-submit with an armed image to get true multimodal fossilization.")
print(f"{'='*64}\n")

# ── 7. Compare against all other fossils ──────────────────────────────────────
all_fossils = sorted(glob.glob(os.path.join(ENCODINGS_DIR, "*.pt")),
                     key=os.path.getmtime, reverse=True)
if len(all_fossils) > 1 and rv is not None:
    print(f"─── Resonance against {len(all_fossils)-1} other fossil(s) ──────────────────────")
    for other_path in all_fossils:
        if other_path == fossil_path:
            continue
        try:
            other = torch.load(other_path, map_location="cpu")
            if 'residue_vector' not in other:
                continue
            other_rv = other['residue_vector'].float().flatten()
            # pad/trim to same length
            n = min(len(rv), len(other_rv))
            cos = torch.dot(rv[:n], other_rv[:n]) / (rv[:n].norm() * other_rv[:n].norm() + 1e-8)
            desc_other = other.get('description', '')[:60]
            fp_other = other.get('image_fingerprint')
            modal = "IMG+TXT" if (fp_other is not None and fp_other.norm().item() > 1e-4) else "TXT-only"
            print(f"  [{modal}] cos={cos.item():+.4f}  '{desc_other}...'")
        except Exception as e:
            pass
    print()
