#  CLI Command Guide

This guide provides the canonical list of commands for interacting with the Gyroidic Sparse Covariance Flux Reasoner system.

## [START] Recommended Entry Points

| Tool | Purpose | Command |
| :--- | :--- | :--- |
| **Unified Backend** | Starts the web UI and API | `python hybrid_backend.py` |
| **Dataset CLI** | Ingests data and trains models | `python dataset_cli.py` |
| **Diegetic Terminal** | CLI-based interaction loop | `python src/ui/diegetic_terminal.py` |

---

## [METRICS] Dataset Ingestion & Training (`dataset_cli.py`)

The `dataset_cli.py` is the primary interface for feeding information into the reasoner.

### 1. Simplified Aliases (Recommended)

| Command | Description |
| :--- | :--- |
| `python dataset_cli.py quick-start` | Automates model creation, IMDB ingestion, and training. |
| `python dataset_cli.py add-wiki --topics "Topic1,Topic2"` | Scrapes Wikipedia topics into the manifold. |
| `python dataset_cli.py train-local --path "./data/"` | Ingests a local directory and begins training. |

### 2. Full Command Set

#### `add-dataset`
Ingests data from external sources.
```bash
python dataset_cli.py add-dataset --name "my_data" --source [huggingface|kaggle|wikipedia|local|url|portal] --path "source_identifier"
```
- `--source portal`: Point to a `.txt` file containing a list of URLs to ingest.
- `--preprocessing`: `text` (default), `image`, `tabular`, `multimodal`.
- `--max-samples`: Limit ingestion count.
- `--mandelbulb`: Enable geometric augmentation.

#### `create-model`
Initializes a new reasoner state.
```bash
python dataset_cli.py create-model --name "my_model" --type temporal --functionals 5
```

#### `setup-training`
Links a model to a dataset for an evolution run.
```bash
python dataset_cli.py setup-training --model "my_model" --dataset "my_data" --epochs 10
```

#### `train`
Begins the evolutionary training loop.
```bash
python dataset_cli.py train --model "my_model" --dataset "my_data"
```

---

##  Web Interfaces & Backends

| Backend | Port | Description |
| :--- | :--- | :--- |
| `hybrid_backend.py` | 8000 | **Preferred**. Serves the Conversational GUI. |
| `diegetic_backend.py` | 8001 | Serves the low-level Diegetic Terminal. |

### Accessing the GUIs:
- **Conversational GUI**: `http://localhost:8000/conversational_web_gui.html`
- **Wikipedia Ingestor**: `http://localhost:8000/wikipedia_trainer.html`
- **Diegetic Terminal**: `http://localhost:8000/diegetic_terminal.html`

---

##  Maintenance & Diagnostics

### List All Resources
```bash
python dataset_cli.py list-all
```

### Manual Firewall Configuration (Windows)
If the backend is not accessible on Port 8000, run as Admin:
```powershell
netsh advfirewall firewall add rule name="Python HTTP Server" dir=in action=allow protocol=TCP localport=8000
```

---

## [BRAIN] Anti-Lobotomy Compliance
All commands in this system adhere to the **Silicon Sovereignty** protocols:
1. No hardcoded primes are used for modular arithmetic.
2. Training uses **Survivorship Pressure**, not goal-oriented loss.
3. Entropy is harvested from hardware jitter (`t_RFC` stalls).

---

## [TELEMETRY] Unified CMD Telemetry Reporting (`hybrid_backend.py`)

[`hybrid_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/hybrid_backend.py) prints a unified 5-section ASCII CMD telemetry report after every `process_text()` turn and exposes on-demand inspection commands:

* **Interactive Text / Chat Commands**: Send `CMD_REPORT` or `STATUS_REPORT` to `process_text` (or `/api/chat`) to return the full formatted 5-section report directly as the response text.
* **REST Endpoints**:
  * `GET http://127.0.0.1:8000/api/cmd_report`
  * `GET http://127.0.0.1:8000/api/telemetry_report`

The 5-section report unifies:
1. **`[I. NON-TELEOLOGICAL NOMENCLATURE & FOUNDATIONAL ONTOLOGY (Old <-> Refined Terms)]`**: Explicit mappings between deprecated teleological vocabulary (`Loss Function`, `Training / Optimization`, `Target / Label`, `Convergence`, `Error / Residual`, `Safety Filter`, `Correctness`, `Reasoning`, `Memory`, `Prediction`, `Output Layer`, `Weight Decay`, `Gradient`, `Validation`, `Stop Condition`) and live runtime metrics (`Selection Pressure`, `Topological Evolution / Assimilation`, `Resonant Attractor / Fossil`, `Phase Alignment / Orbit Closure`, `Chirality / Tension / Warp`, `Spectral Speciation / Fracture`, `Structural Integrity / Cohomology`, `Geodesic Navigation`, `Fossilized Dyads / Cavity State`, `Phase-Coupled Extrapolation`, `Diegetic Larynx`, `Entropic Dissolution`, `Covariant Flux`, `Admissibility Audit`, `Asymptotic Hardening`).
2. **`[II. PRIME RESONANCE LADDER, ZEITGEIST CRT POLYTOPE ROUTER & VETO SUBSPACE]`**: `[PRIME_LADDER]` (levels 5, 8, 13, 19, 29, 37, 43, 67, 71), `[ZEITGEIST]` (`W_N`, `S_N`, `P_N` CRT polytope state, `alpha`, `beta`, `gamma` barycentric coordinates, mode `POLYTOPE_RESONANCE` / `DIRECT_PASSTHROUGH` / `ABORT`), and `[VETO_SUBSPACE]` (`Lattice_E8`, `Mandelbulb_S`, `Hyperbolic_H`, `CalabiYau_CY`, `[LOVE_PROTECTOR]`, `[SOFT_GATES]`).
3. **`[III. QUASICRYSTALS, DUAL-LINEAGE BOULIGAND, SU(2) GAUGE OPTICS & rANS/BB-ANS]`**: `[QUASICRYSTAL]` ($E_8 \to H_4$ 600-cell golden folding, Moire phason strain, 14-mode Einstein monotile `Hat`/`Spectre`), `[BOULIGAND_YVES]` ($2\times 2$ Bloch-Haydock helicoidal projector & Bragg-Pippard pitch $2G_0$), `[BOULIGAND_GEORGES]` (Point-Cross tangent cone & infinite-dimensional $B$-subdifferential), `[SU2_GAUGE_OPTICS]` (Non-Abelian $SU(2)$ Kapitza floor, DFM cocycle $\Delta\Phi_{\text{DFM}}$, Pancharatnam-Berry phase), `[RANS_BBANS_CRT]` (rANS + bits-back ANS over CRT residues), and `[SAGAWA_UEDA_KELLY]` (thermodynamic mutual information + fractional Kelly allocation).
4. **`[IV. UNIFIED ENERGY-BASED LEARNING (EBL) & AFFORDANCE LANDSCAPE]`**: `[EBL_LANDSCAPE]` (`E_structural`, `E_codes`, `E_fenchel_gap`, `E_hessian_8x8`, `E_soliton_heal`, `E_valence_hunger`, `E_total`), `[EBL_AFFORDANCE]` (Boltzmann probability, free energy $F = E_{\text{total}} - T \cdot H_{\text{aff}}$, `ALGORITHMIC` $O(N^3)$ gate, 500ms restless thermal latency guard).
5. **`[V. TREE OF THOUGHT (ToT) 5-BRANCH, WHEEL ALGEBRA HUB (bot=0/0) & GOVERNANCE]`**: `[TOT_5BRANCH]` (Collatz / Diophantine near-miss $2^l \approx 3^k$ & surd sandwich $\sqrt{2}+\sqrt{3}\approx\pi$, Differential telemetry wobble-to-noise veto, Prime-stride UFSC sieve $p=197$, 2D square percolation $p_c=0.592746$, BioArchetypal 8 fixed reflexes), `[ONTOLOGICAL_NULL]` (Hegel self-relation $N=(N\mapsto N)$ & Carlstrom Wheel Algebra $\bot=0/0$), `[VANELLOPE_MANDY]` (Alabama apportionment seat sovereignty & Mandy refusal), `[ALEXANDER_RP4]` ($\mathbb{RP}^4$ transversality vs wild Cantor-pinch tangency), and `[FRACTAL_COXETER]` (LightHorus Coxeter $\{3,3,7\}$ fundamental domain isolation).

