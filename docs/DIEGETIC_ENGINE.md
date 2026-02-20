# Diegetic Physics Engine

> The main runtime backend — 3,637-line HTTP server integrating 30+ sub-modules into a single interactive physics engine.
> **Source**: [`src/ui/diegetic_backend.py`](../src/ui/diegetic_backend.py)

---

## 1. Architecture Overview

```mermaid
graph TB
    subgraph HTTP["RequestHandler (HTTP Server)"]
        GET["GET: /graph, /status,<br/>/training_status, /system2"]
        POST["POST: /process, /train,<br/>/ingest_*, /tabby_*"]
    end
    
    GET --> DPE
    POST --> DPE
    
    subgraph DPE["DiegeticPhysicsEngine"]
        PI["process_input<br/>(820L main pipeline)"]
        FWD["forward<br/>(evolutionary pass)"]
        SAVE["save_state / load_state"]
    end
    
    subgraph Modules["Sub-Module Integration"]
        CAV["ResonanceCavity"]
        LAR["Larynx (char→tensor)"]
        CALM_M["CALM Predictor"]
        KAGH["KAGHBlock"]
        SCCCG["SpeculativeCoprimeGate"]
        ADMM["OperationalADMM"]
        FMF["FractalMetaFunctional"]
        GC["GyroidCovarianceEstimator"]
        GRAPH["GyroidicGraphManager"]
    end
    
    PI --> Modules
    
    subgraph Data["Data Integration"]
        PI_D["PressureIngestor"]
        TF["TextbookFilter"]
        TC["TabbyClient"]
        LDL["LocalDataLoader"]
    end
    
    PI --> Data
```

---

## 2. DiegeticPhysicsEngine

### Initialization (30+ sub-modules)

| Category | Components |
|----------|-----------|
| **Core** | `ResonanceCavity`, `ResonanceLarynx`, `GyroidCovarianceEstimator` |
| **KAGH** | `KAGHBlock`, `HarmonicWaveDecomposition` |
| **ADMM** | `OperationalADMM`, `CALM` |
| **Topology** | `SpeculativeCoprimeGate`, `SpeculativeHomologyEngine`, `GyroidicGraphManager` |
| **Extensions** | `MetaPolytopeMatrioshka`, `QuantumInspiredReasoningState` (optional) |
| **Fractal** | `FractalMetaFunctional` |
| **Data** | `PressureIngestor`, `TextbookFilter`, `TabbyClient`, `LocalDataLoader` |

### process_input Pipeline (820 lines)

The main method processes user text through these stages:

| Stage | Method(s) | Purpose |
|-------|-----------|---------|
| 1. Text → Tensor | `_text_to_tensor` | Polynomial rotating hash (anti-lobotomy) |
| 2. Affordance Gradients | `_compute_affordance_gradients` | Soft detection: code, math, conversation, API |
| 3. Constraint Injection | `_inject_constraint_pressure` | Force incompatible compressions to coexist |
| 4. Cavity Process | `cavity.process()` | Resonance dynamics, memory update |
| 5. KAGH + CALM | KAGH block + CALM veto | System 2 repairs if budget allows |
| 6. Response Generation | `_generate_dyad_aware_response` | Enhanced text via association system |
| 7. Gyroid Violation | `_compute_full_gyroid_violation_score` | Spectral + covariance + topological checks |
| 8. Unfolding Closure | `_perform_unfolding_closure_check` | Hyper-ring, cycle closure, triadic reciprocity |
| 9. Graph Update | `_perform_advanced_topological_analysis` | Betti numbers, persistence, graph connectivity |

### Supporting Methods

| Group | Methods |
|-------|---------|
| **Response** | `_generate_enhanced_response`, `_generate_fallback_response`, `_apply_linguistic_correction` |
| **Conversational** | `_detect_conversational_patterns`, `_extract_conversational_embeddings`, `_attempt_api_content_extraction` |
| **Association** | `_handle_dyad_ingestion`, `_handle_association_learning`, `_enhanced_association_learning` |
| **Topology** | `_compute_betti_numbers`, `_detect_topological_cycles`, `_estimate_manifold_curvature` |
| **System 2** | `_run_advanced_physics` (quantum/polytope if budget allows) |
| **Persistence** | `save_state`, `load_state`, `_repair_tensors` |

---

## 3. EncodingManager

Manages persistent encoding files — each interaction's topological trace is saved as a distinct artifact to prevent "erasing of implication."

| Method | Purpose |
|--------|---------|
| `get_latest_iteration()` | Scan encoding dir for last saved iteration |
| `save_encoding(iteration, text, tensors, metrics)` | Timestamped artifact with structural metrics |

---

## 4. RequestHandler (HTTP API)

### GET Endpoints

| Path | Response |
|------|----------|
| `/` | Serve `conversational_web_gui.html` |
| `/api/status` | Engine state, iteration count, component status |
| `/api/graph` | Graph topology JSON (nodes, edges, metrics) |
| `/api/training_status` | Training progress, log, results |
| `/api/system2` | System 2 (ADMM/CALM) diagnostics |

### POST Endpoints

| Path | Purpose |
|------|---------|
| `/api/process` | Process user input through full pipeline |
| `/api/train` | Launch async training (SpectralStructuralTrainer) |
| `/api/ingest_local` | Ingest local data via LocalDataLoader |
| `/api/tabby_test` | Test TabbyML connection |
| `/api/tabby_complete` | Code completion via TabbyML |
| `/api/tabby_chat` | Chat via TabbyML |
| `/api/tabby_generate_sample` | Generate synthetic training samples |

---

## 5. Related Documentation

| Doc | Connection |
|-----|-----------|
| [GYROID_REASONER.md](GYROID_REASONER.md) | Model architecture the engine wraps |
| [SPECULATIVE_COPRIME_GATE.md](SPECULATIVE_COPRIME_GATE.md) | SCCCG recovery used in pipeline |
| [PHYSICS_ADMM.md](PHYSICS_ADMM.md) | System 2 ADMM + CALM integration |
| [DATA_PIPELINE.md](DATA_PIPELINE.md) | Data sources consumed by the engine |
