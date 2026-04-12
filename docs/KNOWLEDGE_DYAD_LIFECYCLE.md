# Ingestion Lifecycle: Knowledge Dyads — Image, Audio & Text

> This document covers the full lifecycle of multimodal **Knowledge Dyads** as they enter the Diegetic Terminal and propagate through the Gyroidic Manifold.  
> Phase 6.3 adds audio dyads and non-commutative routing — **image and audio can now be armed simultaneously**.

---

## 1. What is a Knowledge Dyad?

A Knowledge Dyad is any pairing of a non-linguistic signal with a linguistic anchor:

| Dyad Type | Signal | Linguistic Anchor |
|---|---|---|
| **Image-Text** | Visual fingerprint (Chebyshev L/Cr/Cb) | Descriptive text |
| **Audio-Text** | Acoustic harmonics (Chebyshev amplitude envelope) | Caption, transcription, or description |
| **Image+Audio-Text** | Both simultaneously | Text conditioned on the merged media bias |

All dyad types share the same ingestion path: Chebyshev projection → `_build_fp_bias()` → `meta_state` update via the `commutativity` routing policy.

---

## 2. Image Dyad — Chebyshev Fingerprint Pipeline

The old 137-dim histogram (`{r:[32], g:[32], b:[32], l:[32], texture, edges:[8]}`) has been retired. The new pipeline matches the audio pipeline exactly:

```
Image file (PNG, JPEG, WebP, …)
    → Canvas 64×64 (in browser)
    → BT.601 L / Cr / Cb decomposition    ← 4096 floats per channel
    → K = clamp(round(√4096 / 4), 5, 32)  ← derived from pixel count; never hardcoded
    → Per channel: Hann-windowed frame energies (K+1 frames)
                 → Chebyshev T_k recurrence (k = 0 … K-1)
                 → Birkhoff row normalisation  (L1 → probability row)
                 → Xorshift32 LSB stochastic rounding (scale=1024)
    → Payload: { chebyshev_degree: K, L:[K], Cr:[K], Cb:[K] }
```

**Why Chebyshev instead of histograms?**

Histograms aggregate colour and cannot distinguish two images with identical colour distributions but different spatial structure. Chebyshev coefficients are **spectral**: T_k at higher k captures higher-frequency variations in the windowed energy envelope. An image of uniform blue sky loads T_0; an image of fine-grained turbulence loads T_{K-1}. The coefficient vector is a faithful topological fingerprint of the image's information density.

**Backend reception** (`fingerprint_proj = nn.Linear(96, dim)`, orthogonal init):

```python
# L, Cr, Cb each padded/truncated to K_IMAGE_MAX=32 → concat → 96 dims
flat = L_coeffs + Cr_coeffs + Cb_coeffs
t = torch.tensor(flat).float()[:96]
fp_bias = fingerprint_proj(t.unsqueeze(0))   # [1, dim]
```

Legacy 137-dim histogram dicts are still accepted via a reshape shim for backward compatibility.

---

## 3. Audio Dyad — Chebyshev Harmonics Pipeline

Audio files (`.m4a`, `.mp4a`, `.wav`, `.ogg`) are decoded in the browser via the Web Audio API. The amplitude envelope is then run through the exact same Chebyshev pipeline:

```
Audio file
    → Web Audio API decode
    → Amplitude envelope (sample-by-sample)
    → K = clamp(round(√sample_count / 64), 5, 32)
    → Hann-windowed frame energies → Chebyshev T_k → Birkhoff → LSB-round
    → Payload: { chebyshev_harmonics:[K], chebyshev_degree:K,
                 duration_s, sample_rate, channel_count }
```

**Backend reception** (`audio_dyad_proj = nn.Linear(64, dim)`, orthogonal init):

```python
harmonics = audio_dyad['chebyshev_harmonics']
t = torch.tensor(harmonics).float()[:64]    # K_AUDIO_MAX=64
audio_bias = audio_dyad_proj(t.unsqueeze(0))  # [1, dim]
```

---

## 4. Simultaneous Image + Audio

Both `active_fingerprint` and `active_audio_dyad` are sent together in every `/interact` payload. The backend `_build_fp_bias()` handles both:

```python
def _build_fp_bias(fp_dict, audio_dict) -> Optional[Tensor]:
    parts = []
    if fp_dict:  parts.append(fingerprint_proj(…))   # image
    if audio_dict: parts.append(audio_dyad_proj(…))  # audio
    if not parts: return None
    return torch.stack(parts).mean(dim=0)             # [1, dim] — average
```

The two biases are **averaged**, giving each modality equal structural weight. If you want image to dominate, send only image; if you want audio to dominate, send only audio. When both are present, they co-influence the manifold equally.

**Example payload (image + audio + text)**:

```json
{
  "text": "describe what you feel",
  "fingerprint": {
    "chebyshev_degree": 16,
    "L":  [0.201, 0.143, …],
    "Cr": [0.501, 0.499, …],
    "Cb": [0.499, 0.503, …]
  },
  "audio_dyad": {
    "chebyshev_harmonics": [0.312, 0.189, 0.044, …],
    "chebyshev_degree": 13,
    "duration_s": 4.2,
    "sample_rate": 44100
  },
  "commutativity": "media_first"
}
```

---

## 5. Non-Commutative Routing — What Changes When You Set DYAD ORDER

The `commutativity` field (set by the `DYAD ORDER:` selector in the header) controls **when** in the `process_input` pipeline the media bias is applied relative to the text embedding.

```
                         ┌──────────────────────┐
                         │   media_first         │
                         │   Apply bias to        │
                         │   meta_state BEFORE    │
            media_bias   │   text_to_tensor()     │
           ┌─────────────►                        │
           │             └──────────────────────┘
           │                        ↓
input_text ┤          1. text_to_tensor() 
           │          2. mimicry
           │          3. forward()  ← text shapes manifold
           │                        ↓
           │             ┌──────────────────────┐
           │             │   text_first          │
           └─────────────►  Apply bias to        │
                         │  meta_state AFTER     │
                         │  forward()            │
                         └──────────────────────┘
```

| Mode | Path-Dependence | Use case |
|---|---|---|
| `symmetric` | None — order irrelevant | Balanced exploration |
| `media_first` | Image/audio writes into the manifold **before** language is parsed. Language then filters an already-coloured latent space. | Visual or acoustic context is primary; text is clarification |
| `text_first` | Language carves the manifold; media then adds spectral texture on top of the language-shaped state. | Text is the frame; image/audio is the tone |

This is grounded in Braid Group non-commutativity: `A ∘ B ≠ B ∘ A` when A is the media projection and B is the text hash operator. Different orderings trace genuinely different paths through the manifold, producing structurally distinct outputs for the same inputs.

---

## 6. Storage and Fossilisation

Every dyad ingestion that passes through `/ingest` triggers a **Persistent Encoding** (`data/encodings/encoding_*.pt`):

1. A `KnowledgeDyad` object is created with `image_fingerprint` (tensor) and `linguistic_description` (string).
2. `DAQUFOperator` / `fossilizer.fossilize(dyad, text_tensor)` writes the dyad to disk.
3. Future confabulation recovery uses these fossilised dyads as gravity wells.

Audio dyads currently ride through `/interact` as live per-request biases. Long-term fossilisation of audio harmonics is a planned extension.

---

## 7. Dyad Recovery — Speculative Coprime Gating

If the system enters a converged or low-entropy state (detected by CALM), the `SpeculativeCoprimeGate` recovers structure via:

1. **Wasserstein OT**: Mass transport toward the fossilised dyad manifold.
2. **Coprime Lock**: Recovery succeeds if the new state satisfies `gcd(w_k, p_k) = 1`.
3. **Generative Rupture**: If Mohr-Coulomb yield pressure is exceeded, a new meaning is synthesised from the image-description or audio-description pair.

---

## 8. Summary

```
User Input (image + audio + text)
    ↓
Chebyshev Fingerprint (JS, identical pipeline for both modalities)
    ↓
/interact POST  { fingerprint, audio_dyad, commutativity }
    ↓
_build_fp_bias() → average image + audio bias → [1, dim]
    ↓
Commutativity Routing
    media_first → bias meta_state → text_to_tensor → forward()
    symmetric   → bias input_tensor → forward()
    text_first  → text_to_tensor → forward() → bias meta_state
    ↓
Tri-State Gate (KNOWN / SEARCH_NEEDED / CONFABULATED)
    ↓ (on CONFABULATED or SEARCH_NEEDED)
DiegeticVisualizer → base64 PNG → browser
    → structural_residues → meta_state (κ=0.05)
    → cheby_self_fingerprint → meta_state (κ=0.02)
    ↓
DAQUF Fossilisation (on /ingest)
```

> [!IMPORTANT]
> A Knowledge Dyad is not a "fact" in a database. It is a **Topological Obstruction** that forces the system's thought-trajectories to curve. The non-commutative routing means the *order* in which you present image, audio, and text is itself a structural signal — not metadata.

---

## 9. Internal Fusion — DataAssociationLayer & Residue Fusion

**Source**: [`src/models/diegetic_heads.py`](../src/models/diegetic_heads.py)

> This section describes what happens *inside* the engine after the Chebyshev fingerprint has biased the manifold. These are two distinct layers: the Chebyshev fingerprint (§2–3) is the **external ingestion path** (raw signal → `meta_state` bias). The `DataAssociationLayer` is the **internal fusion path** (tensors already inside the engine → cross-modal residues → dark matter injection).

When an image and a linguistic description are provided through the terminal side panel, they are treated as a single **Knowledge Dyad** $(\mathcal{I}, \mathcal{L})$.

- **Image Stream ($\mathcal{I}$)**: Projected into a sparse latent vector via the `image_emb` hash.
- **Linguistic Stream ($\mathcal{L}$)**: Projected via the `text_emb` hash.

The dyad enters the `DataAssociationLayer`. Unlike standard multimodal fusion (which often collapses features into a mean), our system performs **Residue Fusion**:

1. **Cross-Modality Torsion**: The system calculates the "shear" between the image features and the text features.
2. **K-Sparse Residue Generation**: The interaction produces $k$ distinct **Residues** ($R_1 \dots R_k$).  
   These residues represent the *incompatibility* between the modalities — what is left over when you try to map a picture to a word.
3. **Resonance Injection**: These residues are injected directly into the `ResonanceCavity` as "Dark Matter" seeds — `D_dark`.

### Relationship to the Chebyshev fingerprint

| Layer | When it runs | What it processes | Where the result goes |
|---|---|---|---|
| Chebyshev fingerprint + `_build_fp_bias()` | Before / during `forward()` | Raw image pixels → K spectral coefficients | `meta_state` (bias) |
| `DataAssociationLayer` residue fusion | Inside `_handle_dyad_ingestion()` | Internal `image_emb` + `text_emb` tensors | `ResonanceCavity` dark matter `D_dark` |

Both layers interact with the manifold, but at different depths and through different channels.

---

## 10. Original §5 Summary of Flow

1. **Terminal**: User provides $(\text{Image} \leftrightarrow \text{Word})$ or $(\text{Audio} \leftrightarrow \text{Word})$.
2. **Chebyshev Fingerprint**: Client-side Chebyshev projection biases `meta_state` via commutativity routing.
3. **Association**: `DataAssociationLayer` computes $k$-residues ($R_k$) from internal `image_emb`/`text_emb`.
4. **Cavity**: $R_k$ warps the dark matter field $D_{dark}$.
5. **Encoding**: State is fossilised to disk (`data/encodings/encoding_*.pt`).
6. **Speculation**: Future "stuck" states use these fossilised dyads as gravity wells to bridge through the vacuum of noise.

