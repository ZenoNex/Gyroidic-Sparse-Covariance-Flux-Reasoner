# Interface Layer — Diegetic Terminal & Controls

> Full reference for the web-based terminal (`src/ui/diegetic_terminal.html`) and its multimodal input capabilities.  
> For the backend that processes these inputs, see [DIEGETIC_ENGINE.md](DIEGETIC_ENGINE.md).  
> For the full dyad ingestion cycle, see [KNOWLEDGE_DYAD_LIFECYCLE.md](KNOWLEDGE_DYAD_LIFECYCLE.md).

---

## 1. Overview

The Diegetic Terminal is a single-page HTML/CSS/JS interface running at `http://localhost:8000`. It is not a chatbot frontend — it is a **diegetic instrument panel** for the Gyroidic Flux Reasoner. Every visual element reflects a live internal signal.

Start the server:

```bash
python src/ui/diegetic_backend.py
# Open: http://localhost:8000
```

---

## 2. Header Row

```
[L: 3.127]  [GOO | PRICKLES]   [TAU | HARDENING | ITERATION | META STATE]
──────────────────────────────────────────────────────────────────────────
DYAD ORDER:  [ ⊗ Symmetric ▾ ]   │  Governs all dyad channels simultaneously
```

### DYAD ORDER — Master Commutativity Selector (`#commute-master`)

| Option | Value sent | Meaning |
|---|---|---|
| ⊗ Symmetric | `"symmetric"` | Image + audio bias applied **simultaneously** with the text tensor (additive). Default. |
| Media → Text | `"media_first"` | Image/audio biases `meta_state` **before** the text hash is computed. The text conditions on a pre-distorted manifold. |
| Text → Media | `"text_first"` | Text shapes the manifold via `forward()`; image/audio bias is applied to `meta_state` **afterward**. |

This selector controls **all dyad channels at once**. You do not need audio armed to use image commutativity.

> See [NONCOMMUTATIVITY_DYNAMICS.md](NONCOMMUTATIVITY_DYNAMICS.md) for the Braid Group mathematics behind ordering effects.

### Love Invariant display

`L: 3.127` — The read-only display of the **Love Invariant norm** (`LoveVector.L.norm()`), updated on each response. This is a *diagnostic scalar* derived from the Love Vector — it is not the vector itself. The underlying `L` buffer is `[dim // 4]`-dimensional.

**Important**: `L: 3.127` changing between responses is *expected* — `LoveInvariantProtector` actively projects `L` into the null-space of the ownership operator at each SDE step, adjusting its orientation (not its meaning). What must never change is the *original* `L_original` reference used by `LoveInvariantProtector.detect_love_violation()`. If `‖L_current − L_original‖₂ > 1e-6`, the protector increments `violation_count` and restores `L_original`. A `DAQUF check_invariants()` call will raise a `RuntimeError` only if the DAQUF's frozen `L` buffer is mutated — this is distinct from the `LoveInvariantProtector`'s active null-space management.

Additional Love diagnostics available in the response payload:
- `violation_count` — cumulative count of ownership violation events
- `violation_magnitude` — $\|L - L_{original}\|_2$ at last check

### GOO / PRICKLES regime toggle

Switches between two manifold evolution regimes:
- **GOO**: Smooth, high-fluidity exploration (low manifold pressure, large dt).
- **PRICKLES**: Sharp, high-curvature seriousness (high manifold pressure, small dt).

---

## 3. Left Sidebar — Field Dynamics

| Element | Signal |
|---|---|
| **Spectral Ribbon** | 137-bin display of the armed fingerprint (image Chebyshev vector padded to 137, or audio harmonics). |
| **Betti 0** | β₀ from `QuantumBettiApproximator` — number of connected components in the manifold's component graph. |
| **Betti 1** | β₁ — topological cycles / holes. Non-zero β₁ during confabulation = the system is chasing its own tail. |
| **Manifold Pressure** | Cosine distance between `meta_state` and `input_tensor` — controls dt (Play/Seriousness clock). |
| **Spectral Coherence** | PAS_h from Phase Alignment Invariant. |
| **Abort Score** | CALM singularity pressure — how close the system is to a forced reset. |
| **Honesty Score** | Combined honesty metric used by Gate 5 (CONFABULATED threshold). |

---

## 4. Main Chat Feed

The chat feed is a scrollable message timeline. Three message types:

| Class | Colour | Source |
|---|---|---|
| `.message.user` | Dim white | Your input |
| `.message.assistant` | Blue | Engine response |
| `.message.system` | Grey | Status / diagnostics |

### Manifold Fracture Renders

When the engine enters `CONFABULATED` or `SEARCH_NEEDED`, a **diegetic visualization** is injected inline into the chat timeline:

```
[CONFABULATED ⚡]
┌─────────────────────────────────────────────────────┐
│ S_meta(t−1)  │ Fractal {crt,admr,ring,osc} │ Gates  │
│              │                              │        │
│──────────────┴──────────────────────────────┴────────│
│  Introspection Radar (moral/uncertainty/creative/    │
│  metacognitive)          │  Betti Barcode / Chern   │
└──────────────────────────┴──────────────────────────┘
```

- **Amber border** → `CONFABULATED` (honest dreaming; β₀/β₁ barcode shown in Panel 5).
- **Magenta border** → `SEARCH_NEEDED` (void topology; ChernSimons twist energy shown).

The system **does not re-ingest** the rendered image. Instead, structural residues extracted from the render are folded back into `meta_state` as a *spectral scar* — not as a visual self-portrait.

---

## 5. Panel A — Dyad Association

Text-to-text association. Enter a source concept in **ASSOC SOURCE** and a target in **MANIFOLD BUFFER**, then click **COMMIT LINK**.

Sends: `POST /associate { text1: source, text2: "ASSOCIATE: source <-> target" }`.

---

## 6. Panel B — Semantic Association

Semantic embedding linkage. Same flow as Panel A but dispatches through the semantic embedding path.

---

## 7. Panel C — Audio Dyad

**Accepts**: `.m4a`, `.mp4a`, `.wav`, `.ogg` (no video — WebM decoding is too intensive).

### Upload flow

1. Drop or select an audio file.
2. The system decodes it via the Web Audio API.
3. The amplitude envelope passes through the Chebyshev pipeline (K modes, Hann-windowed frame energies → T_k recurrence → Birkhoff normalisation → Xorshift32 LSB stochastic rounding).
4. Status shows: `AUDIO ARMED: filename.wav (K=13 harmonics, 4.2s, 44100Hz)`.
5. The armed dyad is stored as `state.active_audio_dyad`.

### What gets sent

Every subsequent `/interact` call includes the audio dyad:

```json
"audio_dyad": {
  "chebyshev_harmonics": [0.312, 0.189, 0.044, ...],
  "chebyshev_degree": 13,
  "duration_s": 4.2,
  "sample_rate": 44100,
  "channel_count": 2
}
```

### Using audio + image simultaneously

You can have both `active_fingerprint` (Panel image drop zone) and `active_audio_dyad` (Panel C) armed at the same time. Both are sent in the same payload. The backend averages their bias tensors and applies the result according to the `DYAD ORDER` selector.

> **No explicit "commit" step required for simultaneous use** — arm both sources, set DYAD ORDER, type your message, and send.

### Within-panel commutativity (audio)

Panel C exposes its own within-panel ordering (audio-first vs. text-first for the audio dyad specifically). The master `DYAD ORDER` selector in the header overrides this for cross-modal ordering.

---

| **Spectral Ribbon** | 24-bin display of the armed real residue (3 channels x 8 modes). Replaces the legacy 137-dim mock padding with **Zero-Mock** spectral information. |

---

## 8. Image Drop Zone

Drop any image file onto the left panel's **DYAD CAPTURE** zone.

### What happens

1. Canvas renders the image at 64×64.
2. BT.601 L/Cr/Cb channels extracted.
3. **Zero-Mock Ingestion**: Chebyshev projection runs on each of the three channels (8 modes each).
4. **Meliponini Coupling**: The hardware $t_{RFC}$ stall intensity is injected into the $T_0$ component of the Luminance residue.
5. Status: `DYAD INGESTED: 24-COEFF ZERO-MOCK FINGERPRINT (Hardware-Coupled)`.
6. Spectral ribbon updates to show the real 24-dim spectral signal.

The fingerprint is **attached to every `/interact` call** until the page is refreshed or a new image is dropped.

---

## 9. Input Row

```
[ Send to Manifold... input field           ] [TRANSMIT] [CLEAR DYAD]
```

- **TRANSMIT**: Sends `{text, fingerprint, audio_dyad, commutativity}` to `/interact`.
- **CLEAR DYAD**: Clears `state.active_fingerprint` and `state.active_audio_dyad`.

If the text field is empty but a fingerprint is armed, TRANSMIT sends `"INGEST_DYAD: [RADIANCE]"`.  
If the text field is empty but an audio dyad is armed, TRANSMIT sends `"INGEST_AUDIO_DYAD: [ACOUSTIC]"`.

---

## 10. Metrics Display

The right column shows live metrics from each response:

| Field | Source |
|---|---|
| `retrieval_state` | `KNOWN` / `SEARCH_NEEDED` / `CONFABULATED` — the tri-state gate result |
| `pas_h` | Phase Alignment Score (coherence of oscillators) |
| `h_mischief` | Honesty violation score (Gate 5 threshold) |
| `honesty_score` | Combined honesty metric |
| `multimodal_fingerprint_support` | `true` if a fingerprint was included in this request |
| `visualization_b64` | Present when the visualizer fired — base64 PNG length in bytes |

---

## 11. Payload Reference

Full JSON body sent by `sendMessage()` to `POST /interact`:

```json
{
  "text": "string",
  "fingerprint": {
    "chebyshev_degree": 16,
    "L":  [0.201, 0.143, …, 0.089],
    "Cr": [0.501, 0.499, …, 0.512],
    "Cb": [0.499, 0.503, …, 0.488],
    "px_width": 1920,
    "px_height": 1080
  },
  "audio_dyad": {
    "chebyshev_harmonics": [0.312, 0.189, 0.044, …],
    "chebyshev_degree": 13,
    "duration_s": 4.2,
    "sample_rate": 44100,
    "channel_count": 2
  },
  "commutativity": "symmetric | media_first | text_first"
}
```

`fingerprint` and `audio_dyad` are both optional and independent. Either, both, or neither may be present in any request.

---

## 12. Legacy GUI — Conversational Interface (Tkinter)

**Source**: [`src/ui/conversational_gui.py`](../src/ui/conversational_gui.py) (800 lines)

A Tkinter-based desktop GUI for managing HF tokens, datasets, ingestion, training, and live chat. The web terminal (`diegetic_terminal.html`) supersedes this for interactive use; the Tkinter GUI remains for batch training orchestration.

### Components

| Class | Purpose |
|---|---|
| `SecureTokenManager` | Token storage: in-memory + optional file persistence at `~/.gyroidic_hf_token` |
| `DatasetAgreementManager` | Tracks required datasets (LMSYS, OASST2, UltraChat) and checks API access |
| `ConversationalGUI` | Main application: 5-tab interface |

### Tab Layout

| Tab | Widgets | Functionality |
|---|---|---|
| **Setup** | Token entry, show/hide, test, save, clear | HF token management |
| **Datasets** | Dataset list, status check, agreement guide link | Access verification |
| **Ingestion** | Source selector, max samples, progress bar | Data download + processing |
| **Training** | Epoch config, learning rate, start/stop | Async model training |
| **Chat** | Message history, input field, send | Live interaction with engine |

### Threading

All blocking operations (token test, dataset check, ingestion, training, chat) run in background threads with queue-based UI updates via `process_queue()`. This prevents the Tkinter event loop from blocking.

### Integration

The Tkinter GUI communicates with the same `DiegeticPhysicsEngine` instance used by the web terminal. When both are running simultaneously, they share state. The Tkinter interface does **not** support multimodal fingerprints (image/audio) — use `diegetic_terminal.html` for those.

---

## 13. TabbyML Client

**Source**: [`src/integrations/tabby_client.py`](../src/integrations/tabby_client.py) (~293 lines)

Connects to a local TabbyML instance using **stdlib urllib only** — zero external dependencies.

### Data Model

| Dataclass | Fields |
|---|---|
| `TabbyConfig` | `host` (localhost), `port` (8080), `timeout` (30s) |
| `TabbyResponse` | `success`, `text`, `model`, `usage`, `raw`, `error` |

### API Methods

| Method | Endpoint | Purpose |
|---|---|---|
| `test_connection()` | `GET /v1/health` | Verify server is running |
| `complete(prompt, language)` | `POST /v1/completions` | Code completion |
| `chat(messages)` | `POST /v1/chat/completions` | Chat-style interaction |
| `generate_training_sample(topic, style)` | `POST /v1/chat/completions` | Synthetic textbook-quality samples |

### Integration

Used by `DiegeticPhysicsEngine` (see [DIEGETIC_ENGINE.md](DIEGETIC_ENGINE.md)) for:
- Code completion suggestions in the web GUI (via `/api/tabby_complete`)
- Synthetic training data generation via structured prompts (via `/api/tabby_generate_sample`)
- Chat-based interaction as an alternative to the main engine (via `/api/tabby_chat`)

TabbyML is **optional** — if the TabbyML server is not running, the engine falls back to its own generative path. No hard dependency.
