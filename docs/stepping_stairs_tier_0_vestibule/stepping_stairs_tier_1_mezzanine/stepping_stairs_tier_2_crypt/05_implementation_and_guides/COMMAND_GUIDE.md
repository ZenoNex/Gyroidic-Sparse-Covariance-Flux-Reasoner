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
