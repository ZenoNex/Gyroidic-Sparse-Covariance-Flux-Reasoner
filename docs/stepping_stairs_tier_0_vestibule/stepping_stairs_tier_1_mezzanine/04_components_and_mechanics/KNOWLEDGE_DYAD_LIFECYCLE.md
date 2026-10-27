# Ingestion Lifecycle: Knowledge Dyads  Image, Audio & Text

> This document covers the full lifecycle of multimodal **Knowledge Dyads** as they enter the Diegetic Terminal and propagate through the Gyroidic Manifold.  
> Phase 6.3 adds audio dyads and non-commutative routing  **image and audio can now be armed simultaneously**.

---

## 1. What is a Knowledge Dyad?

A Knowledge Dyad is any pairing of a non-linguistic signal with a linguistic anchor:

| Dyad Type | Signal | Linguistic Anchor |
|---|---|---|
| **Image-Text** | Visual fingerprint (Chebyshev L/Cr/Cb) | Descriptive text |
| **Audio-Text** | Acoustic harmonics (Chebyshev amplitude envelope) | Caption, transcription, or description |
| **Image+Audio-Text** | Both simultaneously | Text conditioned on the merged media bias |

All dyad types share the same ingestion path: Chebyshev projection  `_build_fp_bias()`  `meta_state` update via the `commutativity` routing policy.

---

## 2. Image Dyad  Chebyshev Fingerprint Pipeline

The old 137-dim histogram (`{r:[32], g:[32], b:[32], l:[32], texture, edges:[8]}`) has been retired. The new pipeline matches the audio pipeline exactly:

```
Image file (PNG, JPEG, WebP, ...)
    --> Canvas 64x64 (in browser or Python)
    --> BT.601 L / Cr / Cb decomposition -> 4096 floats per channel
    --> K = clamp(round(4096 / 4), 5, 32) -> derived from pixel count; never hardcoded
    --> Ingestion Pathways:
        1. Multi-Basis Spectral Decomposition (PolynomialBasis: Chebyshev, Legendre, Hermite, Laguerre, B-Spline)
        2. Chromata Streamline Flowline Tracer (Homotopy filaments with saturation avoidance)
        3. Geometrize Parametric Primitive Decomposer (Closed-form superquadrics & ellipsoids)
    --> Birkhoff row normalisation -> (L1 -> probability row)
    --> Xorshift32 LSB stochastic rounding (scale=1024)
    --> Payload: { basis_type, degree: K, L:[K], Cr:[K], Cb:[K], chromata_filaments, geometrize_primitives }
```

### 2.1 The Multi-Basis Polynomial Spectrum

Moving beyond single-basis representations, [`image_extension.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/image_extension.py) and [`src/ui/diegetic_visualizer.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_visualizer.py) provide multi-basis polynomial decomposition across canonical orthogonal families:

- **Chebyshev ($T_n$)**: Weight $w(x) = (1 - x^2)^{-1/2}$. Minimax equiripple error distribution across the frequency band. Optimal for perceptual hashing and edge-preserving texture frequency analysis.
- **Legendre ($P_n$)**: Weight $w(x) = 1$. Flat $L^2$ uniform energy weighting. Unlike Chebyshev, which clusters evaluation nodes near boundary points ($\pm 1$), Legendre weights interior visual density equally, making it ideal for interior mass and volumetric contrast.
- **Hermite ($He_n$)**: Weight $w(x) = e^{-x^2 / 2}$ on $(-\infty, \infty)$. Matches the optical Gaussian point-spread function (PSF) and human visual fovea (Gabor filters). Provides superior invariance against Gaussian blur, out-of-focus optics, and atmospheric scattering.
- **Laguerre ($L_n$)**: Weight $w(x) = e^{-x}$ on $[0, \infty)$. Evaluated on radial distance $r = \|\mathbf{x} - \mathbf{c}\|$ from the visual center of mass, yielding rotation-invariant radial harmonic signatures.
- **True B-Splines / KAN (Cox-de Boor)**: Compact local support: $B_{i, k}(x) = 0$ outside $[t_i, t_{i+k+1}]$. Unlike global polynomials where local tampering corrupts all coefficients, B-spline coefficients localize image modifications (watermarks, occlusions, inpainting) strictly to local knot intervals.

### 2.2 Chromata Vector Streamlines & Homotopy Filaments

Traditional pixel hashing suffers from rigid Cartesian grid alignment. Inspired by Chromata (Michael Bromley), autonomous multi-agent path finders walk the continuous 2D/3D gradient vector field $\mathbf{v}(x, y) = \nabla I(x, y)$ or Hamiltonian isoline field $\mathbf{v}^\perp$:

$$\frac{d\mathbf{x}_a}{dt} = \mathbf{v}(\mathbf{x}_a)$$

- **Saturation Avoidance**: Agents deposit a traversal footprint in a spatial spatial-hash grid. If an adjacent coordinate is saturated, agents deflect orthogonally ($\mathbf{v}_{\text{avoid}} \perp \mathbf{v}$).
- **Homotopy Separatrices**: Streamlines lock onto the structural ridge lines and critical separatrices of the image topology rather than arbitrary pixel grids.
- **Topological Invariants**: Extracts arc-length spectra, winding numbers $\mathcal{W} = \frac{1}{2\pi} \oint d\theta$, and curvature integrals $\kappa = \int \|\dot{\mathbf{x}} \times \ddot{\mathbf{x}}\| / \|\dot{\mathbf{x}}\|^3 dt$, providing complete affine, scale, and rotation invariance.

### 2.3 Geometrize Parametric Decomposition (Bypassing Voxel Bloat)

Rather than storing raster matrices or volumetric voxel cubes (which suffer from $O(N^3)$ voxel bloat), the Geometrize cleanroom decomposes visual fields into continuous parametric primitives $\mathcal{P} = \{(\mathbf{c}_m, \mathbf{r}_m, \mathbf{q}_m, \mathbf{col}_m, \alpha_m)\}_{m=1}^M$:

- **Parametric Primitives**: Ellipsoids, superquadrics, oriented boxes, and Bezier ribbons.
- **Algebraic Mass-Moment Tensor**:
  $$\mathbf{I} = \sum_{m=1}^M \alpha_m \mathcal{V}_m [(\mathbf{c}_m - \mathbf{\bar{c}})^2 \mathbf{1} - (\mathbf{c}_m - \mathbf{\bar{c}})(\mathbf{c}_m - \mathbf{\bar{c}})^T]$$
- **Principal Eccentricities**: Eigenvalues of $\mathbf{I}$ yield rotation- and scale-invariant geometric eccentricities $(\varepsilon_1, \varepsilon_2)$ and sphericity $\mathcal{S}$.
- **Direct Codec Intersection**: In [`GyroidImageProjector`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/codec/gyroidic_codec.py), continuous primitives algebraically intersect the gyroid minimal surface $G(\mathbf{x}) = 0$, evaluating boundary integrals directly into $GL(n)$ matrix residues without intermediate raster downsampling.

### 2.4 Comparative Fingerprinting Matrix

| Fingerprint Paradigm | Mathematical Domain | Affine Invariance | Occlusion Tolerance | Topological Scars | Computational Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **aHash (Average)** | Discrete Luminance Grid | None (Fragile) | Poor ($< 5\%$) | None (Scalar Mean) | 64 bits (Trivial) |
| **dHash (Difference)** | Discrete Adjacent Gradients | None (Shift sensitive) | Poor ($< 10\%$) | None (Binary Mask) | 64 bits (Trivial) |
| **pHash (DCT)** | Discrete Cosine Spectral | Moderate (Scale only) | Moderate ($< 20\%$) | Weak (Global Spectral) | 64 bits (Low) |
| **wHash (Wavelet)** | Multi-Resolution DWT | Moderate | Moderate ($< 25\%$) | Sub-band Energy | 64-128 bits |
| **Chebyshev ($T_k$)** | Minimax Orthogonal Poly | High (Windowed) | Good ($< 35\%$) | High-Frequency Scar | 96 floats (548 bytes) |
| **Legendre / Hermite** | Flat $L^2$ / Gaussian Kernel | High | High ($< 40\%$) | PSF / Radial Gaussian | 96 floats (548 bytes) |
| **B-Spline (Cox-de Boor)**| Localized Knot Splines | High | Exceptional ($< 60\%$) | Localized Knot Scars | $K \times \text{knots}$ |
| **Chromata Streamline**| Continuous Gradient Vector | Invariant (Continuous)| Exceptional ($< 70\%$) | Homotopy Filaments | 16-bin Filament + Winding |
| **Geometrize Primitive**| Continuous Superquadrics | Full Invariance | Exceptional ($< 75\%$) | Analytical Mass Tensors | $M$ Parametric Primitives |

**Backend reception** (`fingerprint_proj = nn.Linear(96, dim)`, orthogonal init):

```python
# Ingestion via canonical ImageProcessor or direct visualizer extraction
from image_extension import ImageProcessor
from src.codec.vision_utilities import project_multimodal_fingerprint_to_96

proc = ImageProcessor(device='cpu')
# Generates multi-basis, Chromata filaments, or Geometrize primitives
fp = proc.extract_spectral_fingerprint(image_input, k=32, basis_type='chebyshev')
t96 = project_multimodal_fingerprint_to_96(fp)   # [96] canonical tensor
fp_bias = fingerprint_proj(t96.unsqueeze(0))     # [1, dim]
```

Legacy 137-dim histogram dicts and structured Chromata/Geometrize dicts are natively resolved by [`GyroidicCodec.encode()`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/codec/gyroidic_codec.py#L852-L920).

---

## 3. Audio Dyad  Chebyshev Harmonics Pipeline

Audio files (`.m4a`, `.mp4a`, `.wav`, `.ogg`) are decoded in the browser via the Web Audio API. The amplitude envelope is then run through the exact same Chebyshev pipeline:

```
Audio file
     Web Audio API decode
     Amplitude envelope (sample-by-sample)
     K = clamp(round(sample_count / 64), 5, 32)
     Hann-windowed frame energies  Chebyshev T_k  Birkhoff  LSB-round
     Payload: { chebyshev_harmonics:[K], chebyshev_degree:K,
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
    if fp_dict:  parts.append(fingerprint_proj())   # image
    if audio_dict: parts.append(audio_dyad_proj())  # audio
    if not parts: return None
    return torch.stack(parts).mean(dim=0)             # [1, dim]  average
```

The two biases are **averaged**, giving each modality equal structural weight. If you want image to dominate, send only image; if you want audio to dominate, send only audio. When both are present, they co-influence the manifold equally.

**Example payload (image + audio + text)**:

```json
{
  "text": "describe what you feel",
  "fingerprint": {
    "chebyshev_degree": 16,
    "L":  [0.201, 0.143, ],
    "Cr": [0.501, 0.499, ],
    "Cb": [0.499, 0.503, ]
  },
  "audio_dyad": {
    "chebyshev_harmonics": [0.312, 0.189, 0.044, ],
    "chebyshev_degree": 13,
    "duration_s": 4.2,
    "sample_rate": 44100
  },
  "commutativity": "media_first"
}
```

---

## 5. Non-Commutative Routing  What Changes When You Set DYAD ORDER

The `commutativity` field (set by the `DYAD ORDER:` selector in the header) controls **when** in the `process_input` pipeline the media bias is applied relative to the text embedding.

```
                         
                            media_first         
                            Apply bias to        
                            meta_state BEFORE    
            media_bias      text_to_tensor()     
                                   
                        
                                   
input_text           1. text_to_tensor() 
                     2. mimicry
                     3. forward()   text shapes manifold
                                   
                        
                           text_first          
             Apply bias to        
                           meta_state AFTER     
                           forward()            
                         
```

| Mode | Path-Dependence | Use case |
|---|---|---|
| `symmetric` | None  order irrelevant | Balanced exploration |
| `media_first` | Image/audio writes into the manifold **before** language is parsed. Language then filters an already-coloured latent space. | Visual or acoustic context is primary; text is clarification |
| `text_first` | Language carves the manifold; media then adds spectral texture on top of the language-shaped state. | Text is the frame; image/audio is the tone |

This is grounded in Braid Group non-commutativity: `A  B  B  A` when A is the media projection and B is the text hash operator. Different orderings trace genuinely different paths through the manifold, producing structurally distinct outputs for the same inputs.

---

## 6. Storage and Fossilisation

Every dyad ingestion that passes through `/ingest` triggers a **Persistent Encoding** (`data/encodings/encoding_*.pt`):

1. A `KnowledgeDyad` object is created with a `linguistic_description` (string), and may include `image_fingerprint`, `audio_harmonics`, or `unified_spectral_signature` tensors.
2. `DAQUFOperator` / `fossilizer.fossilize(dyad, text_tensor)` detaches all tensor attributes to CPU and serializes them to disk.
3. **Index resumptions**: To avoid $O(N)$ startup scan bottlenecks on large datasets, `DyadFossilizer` initializes instantly by reading the fast `.fossil_index.json` snapshot index.
4. **Extractable Agent Protocol**: A dyad that reaches a stable GLYPHLOCK state can be cleanly exported via `fossilizer.export_agent_smith()`. This generates a `soliton_smith.pt` file containing only the structural algebra (prime frequencies, Betti numbers, symbolic residues), completely decoupling the "Syntax" from the hardware "Substrate".
5. Future confabulation recovery uses these fossilised dyads as gravity wells.

Audio dyads, image fingerprints, and Video base64 streams are formally integrated. Video streams are parsed by `VideoDyadParser` with a relaxed temporal covariance sparsification threshold of `0.50` (retaining 50% of the variance outlier maps to prevent over-compressing frame continuity). Audio inputs map to `audio_harmonics` tensors, and unified spectrums map to `unified_spectral_signature`. All tensors are safely detached/cpu-copied before serialization.

### 6.1 Internal Ouroboros Dyad Generation
Knowledge Dyads are not strictly limited to external signals provided by the user. Through the **Ouroboros Loop** (Phase 6.3 Update), the system autonomously generates internal dyads.
*   **Shadow Trapping**: `UniversalOrchestrator` and `SpeculativeCoprimeGate` independently log structural failures, mathematical paradoxes, and non-commutative collapses into their `[SHADOW LOG]` buffer.
*   **Self-Ingestion**: At the end of every inference loop, the `DiegeticEngine` queries these logs via `pop_shadow_logs()`. 
*   **Fossilization**: Each caught anomaly is immediately wrapped in a `KnowledgeDyad`, bound to the system's exact topological state coordinates, and sent to `DyadFossilizer.fossilize()`. This loop forces the system to literally *digest* its failures as topological anchors.

---

## 7. Dyad Recovery  Speculative Coprime Gating

If the system enters a converged or low-entropy state (detected by CALM), the `SpeculativeCoprimeGate` recovers structure via:

1. **Wasserstein OT**: Mass transport toward the fossilised dyad manifold.
2. **Coprime Lock**: Recovery succeeds if the new state satisfies `gcd(w_k, p_k) = 1`.
3. **Generative Rupture**: If Mohr-Coulomb yield pressure is exceeded, a new meaning is synthesised from the image-description or audio-description pair.

---

## 8. Summary

```
User Input (image + audio + text)
    
Chebyshev Fingerprint (JS, identical pipeline for both modalities)
    
/interact POST  { fingerprint, audio_dyad, commutativity }
    
_build_fp_bias()  average image + audio bias  [1, dim]
    
Commutativity Routing
    media_first  bias meta_state  text_to_tensor  forward()
    symmetric    bias input_tensor  forward()
    text_first   text_to_tensor  forward()  bias meta_state
    
Tri-State Gate (KNOWN / SEARCH_NEEDED / CONFABULATED)
     (on CONFABULATED or SEARCH_NEEDED)
DiegeticVisualizer  base64 PNG  browser
     structural_residues  meta_state (=0.05)
     cheby_self_fingerprint  meta_state (=0.02)
    
DAQUF Fossilisation (on /ingest)
```

> [!IMPORTANT]
> A Knowledge Dyad is not a "fact" in a database. It is a **Topological Obstruction** that forces the system's thought-trajectories to curve. The non-commutative routing means the *order* in which you present image, audio, and text is itself a structural signal  not metadata.

---

## 9. Internal Fusion  DataAssociationLayer & Residue Fusion

**Source**: [`src/models/diegetic_heads.py`](../../../../src/models/diegetic_heads.py)

> This section describes what happens *inside* the engine after the Chebyshev fingerprint has biased the manifold. These are two distinct layers: the Chebyshev fingerprint (23) is the **external ingestion path** (raw signal  `meta_state` bias). The `DataAssociationLayer` is the **internal fusion path** (tensors already inside the engine  cross-modal residues  dark matter injection).

When an image and a linguistic description are provided through the terminal side panel, they are treated as a single **Knowledge Dyad** $(\mathcal{I}, \mathcal{L})$.

- **Image Stream ($\mathcal{I}$)**: Projected into a sparse latent vector via the `image_emb` hash.
- **Linguistic Stream ($\mathcal{L}$)**: Projected via the `text_emb` hash.

The dyad enters the `DataAssociationLayer`. Unlike standard multimodal fusion (which often collapses features into a mean), our system performs **Residue Fusion**:

1. **Cross-Modality Torsion**: The system calculates the "shear" between the image features and the text features.
2. **K-Sparse Residue Generation**: The interaction produces $k$ distinct **Residues** ($R_1 \dots R_k$).  
   These residues represent the *incompatibility* between the modalities  what is left over when you try to map a picture to a word.
3. **Resonance Injection**: These residues are injected directly into the `ResonanceCavity` as "Dark Matter" seeds  `D_dark`.

### Relationship to the Chebyshev fingerprint

| Layer | When it runs | What it processes | Where the result goes |
|---|---|---|---|
| Chebyshev fingerprint + `_build_fp_bias()` | Before / during `forward()` | Raw image pixels  K spectral coefficients | `meta_state` (bias) |
| `DataAssociationLayer` residue fusion | Inside `_handle_dyad_ingestion()` | Internal `image_emb` + `text_emb` tensors | `ResonanceCavity` dark matter `D_dark` |

Both layers interact with the manifold, but at different depths and through different channels.

---

## 10. Original 5 Summary of Flow

1. **Terminal**: User provides $(\text{Image} \leftrightarrow \text{Word})$ or $(\text{Audio} \leftrightarrow \text{Word})$.
2. **Chebyshev Fingerprint**: Client-side Chebyshev projection biases `meta_state` via commutativity routing.
3. **Association**: `DataAssociationLayer` computes $k$-residues ($R_k$) from internal `image_emb`/`text_emb`.
4. **Cavity**: $R_k$ warps the dark matter field $D_{dark}$.
5. **Encoding**: State is fossilised to disk (`data/encodings/encoding_*.pt`).
6. **Speculation**: Future "stuck" states use these fossilised dyads as gravity wells to bridge through the vacuum of noise.

---

## 11. Compute Affordance Scaling & TailSlayer PyOpenCL Acceleration

Ingesting, decomposing, and fossilizing multi-modal knowledge dyads generates significant computational load. The pipeline dynamically self-regulates via [`HardwareMonitor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py#L4):

### 11.1 Affordance-Driven Spectral Scaling
- **Full Bandwidth (`G_silicon > 0.6` and `Margin_thermal >= 0.25`)**:
  Extracts complete multi-basis orthogonal decompositions (Chebyshev $T_n$, Legendre $P_n$, Hermite $He_n$, B-Splines) at maximum degree $K=32$ with full Chromata streamline flowline tracing.
- **Throttled Bandwidth (`0.25 <= G_silicon <= 0.6` or `0.05 <= Margin_thermal < 0.25`)**:
  Damps polynomial degree to $K=8$ and falls back from continuous Chromata tracing to discrete parametric Geometrize primitive evaluation.
- **Hard Runaway Veto (`Margin_thermal < 0.05` or `thermal_runaway == True`)**:
  Suppresses real-time multi-modal ingestion. Heavy Agent Smith disk serialization is buffered into memory, preventing $O(N)$ disk I/O contention and thermal melting.

### 11.2 TailSlayer PyOpenCL Zero-Copy Hardware Acceleration
When [`SiliconSovereigntyEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/pyopencl_sovereignty.py#L80) detects an active OpenCL context, visual and acoustic multi-basis convolutions are offloaded to **TailSlayer Queue B**:
- Executes zero-copy matrix contractions directly in GPU memory.
- Bypasses PyTorch autograd graph overhead by evaluating Chebyshev recurrence relations ($T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$) as a native OpenCL SIMD kernel.
- Leverages the TailSlayer Bypass: if the host CPU completes the linguistic embedding before the GPU finishes the visual projection, the first-to-finish result immediately seeds the reasoner's forward state.

---

## 12. Interactive Topology: DaemonAton Generative Mind Map Integration

To inspect, explore, and navigate the expanding manifold of fossilized Knowledge Dyads in real time, the reasoner integrates the **DaemonAton Generative Mind Map**:
- **Interface**: Implemented in [`src/ui/generative_mind_map.html`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/generative_mind_map.html) and served live at `http://localhost:8080/mindmap`.
- **D3 Radial Separation Engine**: Utilizes DaemonAton's radial tree spacing equation:
  $$\text{separation}(a, b) = \text{baseSeparation} \cdot \max(1, 10 - d_a) \cdot \text{outerSpacing} \cdot 2^{d_a}$$
  preventing visual node occlusion across deeply nested dyad hierarchies.
- **Live Dyad Feed**: Connected to the backend `/api/mindmap-data` endpoint in [`src/ui/diegetic_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py#L7560), streaming active Poincaré fossils from [`DyadFossilizer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/knowledge_dyad_fossilizer.py#L22), archetype balances, and compute affordance gauges into the interactive matrix tree.


