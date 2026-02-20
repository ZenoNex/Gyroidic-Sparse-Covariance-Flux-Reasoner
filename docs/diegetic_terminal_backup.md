# Diegetic Terminal Backup Reference

> [!CAUTION]
> **ANTI-DELETION BACKUP**: This document preserves the required functionality for `diegetic_terminal.html`. 
> If the HTML file is ever deleted or corrupted, use this document to reconstruct it.

---

## Purpose

The Diegetic Terminal (`src/ui/diegetic_terminal.html`) is the primary human-to-system interface for the Gyroidic Sparse Covariance Flux Reasoner. It provides:

1. **Conversational Interface** - Chat with the Synthetic Larynx
2. **Knowledge Dyad Ingestion** - Text-to-Image association mapping
3. **Association Creation** - Text-to-Text / Source → Target concept linking
4. **Field Dynamics Visualization** - Real-time manifold state monitoring

---

## Backend API Endpoints

The terminal communicates with `diegetic_backend.py` (runs on `http://localhost:8000`):

### `POST /interact`
Primary conversational endpoint.

**Request:**
```json
{
  "text": "User message here"
}
```

**Response:**
```json
{
  "response": "Generated text from Synthetic Larynx",
  "iteration": 42,
  "chiral_score": 0.876,
  "pas_h": 0.923,
  "trust_scalars": [0.8, 0.9, 0.7, 0.85, 0.75],
  "spectral_diagnostics": {...},
  "repair_diagnostics": {...},
  "affordance_gradients": {...}
}
```

### `POST /associate`
Text-to-text association (Source Concept ↔ Target Content).

**Request:**
```json
{
  "source": "Concept name",
  "target": "Associated content or reasoning"
}
```

**Response:**
```json
{
  "status": "associated",
  "source": "Concept name",
  "target": "Associated content",
  "metrics": {...}
}
```

### `POST /ingest`
Knowledge Dyad ingestion (Text + Image Fingerprint).

**Fingerprint Structure (137-dimensional vector):**
| Component | Dimensions | Description |
|-----------|------------|-------------|
| R Histogram | 32 bins | Red channel intensity (0-255 → 32 bins, normalized 0-1) |
| G Histogram | 32 bins | Green channel intensity |
| B Histogram | 32 bins | Blue channel intensity |
| L Histogram | 32 bins | Luminance (avg RGB → 32 bins) |
| Texture | 1 value | Variance-based complexity (0=smooth, 1=complex) |
| Edge Features | 8 values | Sobel directional edges (N, NE, E, SE, S, SW, W, NW) |

**Request:**
```json
{
  "label": "Text label for the dyad",
  "fingerprint": {
    "r": [0.03, 0.05, ...],  // 32 normalized histogram values
    "g": [0.02, 0.04, ...],  // 32 values
    "b": [0.01, 0.06, ...],  // 32 values
    "l": [0.02, 0.03, ...],  // 32 luminance values
    "texture": 0.45,         // complexity score
    "edges": [0.1, 0.2, 0.15, 0.08, 0.12, 0.18, 0.09, 0.11]  // 8 directions
  }
}
```

**Backend Processing:**
- Flattened to 137-dim tensor
- Projected via `nn.Linear(137, 64)` with orthogonal initialization
- Added to input tensor with 0.5 weight

**Response:**
```json
{
  "status": "ingested",
  "label": "Text label",
  "metrics": {...}
}
```



### `GET /graph`
Returns topological graph data for visualization.

### `GET /health`
Health check endpoint.

---

## Required UI Components

### 1. Chat Interface
- **Input**: Text field for user messages
- **Output**: Display area for Synthetic Larynx responses
- **Metrics Display**: PAS_h, Chiral Score, Trust Scalars
- **Must call**: `POST /interact`
- **Must show**: Actual generated response text, NOT just diagnostics

### 2. Knowledge Dyad Panel
- **Image Input**: File upload or canvas for drawing
- **Fingerprint Extraction**: Convert image to histogram features
- **Label Input**: Text description for the visual
- **Submit**: Calls `POST /ingest`
- **Flow**: Terminal → DataAssociationLayer → ResonanceCavity → Persistent Encoding

### 3. Association Panel  
- **Source Input**: Concept name
- **Target Input**: Content/reasoning to associate
- **Submit**: Calls `POST /associate`
- **Display**: Confirmation with metrics

### 4. Field Dynamics Panel
- **Manifold Pressure**: Play (high dt) vs Seriousness (low dt)
- **CALM Assessment**: Abort score, trajectory status
- **Spectral Coherence**: From repair system
- **Matrioshka Level**: If quantum extensions active

---

## Core Concepts (from docs/PHILOSOPHY.md)

### Synthetic Larynx
The `ResonanceLarynx` (`src/models/diegetic_heads.py`) generates emergent text responses:
- Operates on character-level ASCII (vocab_size=128)
- Uses topological state from ResonanceCavity
- Responses emerge from field dynamics, not prompted templates

### Resonance Cavity
Memory system storing:
- Pattern representations
- Trust scalars for functional heads
- Dark matter field (latent unknowledge)
- Temporal coherence across interactions

### Knowledge Dyads
Image-Description pairs that create **Topological Obstructions**:
- Force thought-trajectories to curve
- Ensure "No Erasing of Implication"
- Stored as persistent encodings in `data/encodings/`

### Field Dynamics
- **Selection Pressure (S)**: Internal symbolic coherence
- **Containment Pressure (C)**: Physical/topological admissibility
- **Play vs Seriousness**: Time dilation based on manifold curvature

---

## Running the System

```bash
# Start backend
cd "D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner"
conda activate langflow
python -u -X utf8 src/ui/diegetic_backend.py

# Open in browser
http://localhost:8000/
```

---

## File Dependencies

- `src/ui/diegetic_backend.py` - HTTP server and ENGINE
- `src/models/resonance_cavity.py` - Memory and pattern storage
- `src/models/diegetic_heads.py` - ResonanceLarynx, DataAssociationLayer
- `src/core/spectral_coherence_repair.py` - Repair system
- `src/core/chern_simons_gasket.py` - Logic leak prevention
- `src/topology/embedding_graph.py` - Graph visualization

---

## Version History

- **Created**: 2026-02-04 by Claude (Antigravity)
- **Purpose**: Backup reference after content deletion
- **Related Conversation**: 139fd664-6390-4b08-a9be-8a435d630e80
