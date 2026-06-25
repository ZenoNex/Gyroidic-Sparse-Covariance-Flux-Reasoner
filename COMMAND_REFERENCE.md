# Gyroidic AI System - Command Reference

**Quick reference for all command-line interfaces**

---


### Quick Training
```bash
# Train on movie reviews (5 minutes)
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3

# Learn physics from Wikipedia (10 minutes)
python dataset_command_interface.py add-wikipedia --topics physics --samples 300 --train

# Train on your documents
python dataset_command_interface.py train-local --path ./documents/ --epochs 5
```

### System Health
```bash
# Check everything is working
python test_fixes_verification.py

# Check system status
python dataset_command_interface.py status
```

---

## [DOCS] Dataset Training Commands

### Main Interface: `dataset_command_interface.py`

#### Quick Start (Easiest)
```bash
python dataset_command_interface.py quick-start --dataset [name] --samples [N] --epochs [N] [--augment]
```

**Popular Datasets:**
- `imdb` - Movie reviews
- `squad` - Question-answering
- `wikitext` - Wikipedia articles
- `arxiv` - Scientific papers
- `codeparrot` - Programming code

**Examples:**
```bash
python dataset_command_interface.py quick-start --dataset imdb --samples 1000 --epochs 5
python dataset_command_interface.py quick-start --dataset squad --samples 500 --epochs 3 --augment
```

#### Wikipedia Learning
```bash
python dataset_command_interface.py add-wikipedia --topics [topic] --samples [N] [--train]
```

**Topic Collections:**
- `physics` - Quantum mechanics, relativity, thermodynamics
- `mathematics` - Linear algebra, calculus, topology
- `computer_science` - Machine learning, algorithms
- `philosophy` - Philosophy of mind, logic, ethics
- `biology` - Molecular biology, evolution, genetics

**Examples:**
```bash
python dataset_command_interface.py add-wikipedia --topics physics --samples 500 --train
python dataset_command_interface.py add-wikipedia --topics "Quantum_mechanics,Relativity" --samples 300 --train
```

#### Local Files
```bash
python dataset_command_interface.py train-local --path [path] --epochs [N] [--augment]
```

**Examples:**
```bash
python dataset_command_interface.py train-local --path ./documents/ --epochs 10
python dataset_command_interface.py train-local --path ./my_book.txt --epochs 5 --augment
```

#### Full Pipeline (Advanced)
```bash
python dataset_command_interface.py full-pipeline --source [type] --dataset [name] --epochs [N] [--augment]
```

**Sources:** `huggingface`, `wikipedia`, `local`, `portal`

**Examples:**
```bash
python dataset_command_interface.py full-pipeline --source huggingface --dataset squad --augment --epochs 20
python dataset_command_interface.py full-pipeline --source wikipedia --dataset "physics,math" --augment --epochs 25
```

#### System Management
```bash
# List available datasets
python dataset_command_interface.py list-datasets

# Check system status
python dataset_command_interface.py status
```

---

##  Web Interfaces

### Chat Interface
(Served automatically via the unified hybrid backend)
```bash
# Start backend server
.venv\scripts\python.exe hybrid_backend.py

# Open browser to: http://localhost:8000
```

**Features:**
- Chat with AI
- Upload and analyze images
- View system status
- Real-time responses

### Diegetic Terminal Exclusive Chat Commands
When using the chat interface, you can type special command prefixes directly into the prompt text area. These command structures bypass the standard conversation pipeline to run specific system and database tasks:

* **`SOVEREIGN_FETCH:`**
  Manual Sovereign Nutrient Fetch. Pulls high-entropy conversations from StackExchange and HackerNews into the training pipeline.
  * *Syntax*: `SOVEREIGN_FETCH:`
* **`CLOUD_FETCH:`**
  Manual Cloud Nutrient Sync. Synchronizes text and logic shards from the Google Drive cloud workspace.
  * *Syntax*: `CLOUD_FETCH:`
* **`EXPORT_AGENT_SMITH:`**
  Agent Smith Export Protocol. Decouples the active mathematical state and psychological profile (meta-state geometry, Betti numbers, prime frequencies, and gauge fields) and exports them to a `.pt` identity file.
  * *Syntax*: `EXPORT_AGENT_SMITH: [Description/Label]`
* **`IMPORT_AGENT_SMITH:`**
  Agent Smith Import Protocol. Inject and rehydrate an exported soliton identity, aligning the geometry and re-stabilizing the archetypal configuration.
  * *Syntax*: `IMPORT_AGENT_SMITH: [Absolute or relative path to .pt file]`
* **`INGEST_DYAD:`**
  Ingests a concept description or a pair of concept descriptions directly into the active manifold.
  * *Syntax*: `INGEST_DYAD: [Description]` or `INGEST_DYAD: Description A <-> Description B`
* **`ASSOCIATE:`**
  Maps a conceptual association between two ideas, updating the topological coordinate system.
  * *Syntax*: `ASSOCIATE: Concept A <-> Concept B`
* **`INGEST_AUDIO_DYAD:`**
  Ingests base64-encoded audio alongside a linguistic description.
  * *Syntax*: `INGEST_AUDIO_DYAD: [Description]`
* **`INGEST_VIDEO_DYAD:`**
  Ingests base64-encoded video frames alongside a linguistic description.
  * *Syntax*: `INGEST_VIDEO_DYAD: [Description]`

### Wikipedia Trainer
(Served automatically via the unified hybrid backend)
```bash
# Start backend server
.venv\scripts\python.exe hybrid_backend.py

# Open browser to: http://localhost:8000/wikipedia-trainer
```

**Features:**
- Search Wikipedia articles
- Download and process content
- Train on specific topics
- Monitor progress

---

## [TEST] Testing & Verification Commands

### Core System Mathematical & Logic Tests
Run the core mathematical systems verification suite (Krylov subspaces, Bouligand slides, Birkhoff projections, etc.):
```bash
# Direct runner (with built-in thread timeout harness)
$env:PYTHONPATH="."
.venv\scripts\python.exe -u tests/test_core_systems.py

# Pytest runner (runs all cases to completion)
.venv\scripts\python.exe -m pytest -v -o python_functions="_test_*" tests/test_core_systems.py
```

### Silicon Sovereignty (GPU/OpenCL) & Synthesis Tests
Run the GPU acceleration tests (stochastic rounding, dual command queues, Born rule, and Betti number approximation):
```bash
# Verify PyOpenCL kernels and device execution on GPU
$env:PYTHONPATH="."
.venv\scripts\python.exe -m pytest -v tests/test_unicorn_synthesis.py
```

### Algebraic Invariant Verification
```bash
$env:PYTHONPATH="."
.venv\scripts\python.exe -m pytest -v tests/test_algebraic_invariants.py
```

### GPU / OpenCL Device Selection Overrides
To target specific discrete graphics cards (e.g., GTX 1050 Ti, NVIDIA) or fallback platforms, set the environment variables before running the tests:
```powershell
# Target discrete NVIDIA GPU (recommended)
$env:GPU_INDEX="nvidia"
.venv\scripts\python.exe -m pytest -v tests/test_unicorn_synthesis.py

# Explicit device index mapping (0, 1, etc.)
$env:Sovereign_GPU_Index="0"
$env:PYOPENCL_DEVICE_INDEX="1"
.venv\scripts\python.exe -m pytest -v tests/test_unicorn_synthesis.py
```

### General Bug Fixes and Image Verification
```bash
# Test bug fixes (run this first)
.venv\scripts\python.exe test_fixes_verification.py

# Test image processing
.venv\scripts\python.exe test_image_simple.py
.venv\scripts\python.exe test_image_integration.py

# Test Wikipedia system
.venv\scripts\python.exe test_enhanced_wikipedia_system.py
```

### Legacy & Component Level Tests
```bash
# Test association learning
.venv\scripts\python.exe test_enhanced_association_learning.py

# Test dataset augmentation
.venv\scripts\python.exe test_mandelbulb_augmentation.py

# Test repair systems
.venv\scripts\python.exe test_repair_integration.py

# Test Phase 3 (dyad system)
.venv\scripts\python.exe test_phase3_dyad_system.py

# Test Phase 4 (advanced features)
.venv\scripts\python.exe test_phase4_advanced_features.py

# Test enhanced fingerprints
.venv\scripts\python.exe test_enhanced_fingerprint.py
```

---

## [INIT] Configuration Options

### Common Options
```bash
--samples N          # Number of examples (500-5000)
--epochs N           # Training rounds (3-20)
--augment           # Use data expansion
--manifold-aware    # Enable Thick Ingestion (attach manifold residues)
--train             # Auto-start training
```

### Advanced Options
```bash
--functionals N     # Reasoning components (3-8)
--hidden-dim N      # Model size (256, 512, 768)
--batch-size N      # Training batch size (2-8)
--learning-rate F   # Learning speed (1e-5 to 1e-3)
--checkpoint        # Save training progress
```

---

## [METRICS] Quick Reference Tables

### Dataset Sizes (Storage per 1000 samples)
| Type | Size | Good For |
|------|------|----------|
| Text | ~90-160 MB | Most training |
| Images | ~200-400 MB | Visual learning |
| Augmented | +50% | Small datasets |

### Training Times (approximate)
| Samples | Epochs | Time |
|---------|--------|------|
| 500 | 3 | 5-10 min |
| 1000 | 5 | 15-25 min |
| 2000 | 8 | 30-60 min |
| 5000 | 10 | 2-4 hours |

### Recommended Settings
| Use Case | Samples | Epochs | Augment |
|----------|---------|--------|---------|
| Quick test | 500 | 3 | No |
| Good results | 1000-2000 | 5-8 | Yes |
| Best quality | 3000+ | 10+ | Yes |

---

##  Troubleshooting Quick Fixes

### Common Errors
```bash
# PIL version error
pip install --upgrade Pillow

# Type handling error (should be fixed)
python test_fixes_verification.py

# Backend connection lost (Restart the unified server)
.venv\scripts\python.exe hybrid_backend.py

# Out of storage
python dataset_command_interface.py status
# Use smaller --samples values
```

### Performance Issues
```bash
# Training too slow
# Use: --samples 500 --epochs 3

# Out of memory
# Use: --batch-size 2

# Storage full
# Check: python dataset_command_interface.py status
# Clean up old datasets in datasets/ folder
```

---

##  File Locations

### Main Scripts
- `hybrid_backend.py` - Unified backend server (recommended runner)
- `dataset_command_interface.py` - Main dataset training
- `src/ui/diegetic_terminal.py` - Legacy web chat interface script
- `src/ui/diegetic_backend.py` - Legacy backend server script
- `image_extension.py` - Image processing

### Test Scripts
- `test_fixes_verification.py` - Core functionality test
- `test_image_simple.py` - Image processing test
- `test_enhanced_wikipedia_system.py` - Wikipedia test

### Data Directories
- `datasets/` - Downloaded and processed datasets
- `data/encodings/` - Trained model data
- `docs/` - Documentation

---

## [GOAL] Workflow Examples

### Beginner Workflow
```bash
# 1. Test system
python test_fixes_verification.py

# 2. Start unified hybrid backend server
.venv\scripts\python.exe hybrid_backend.py
# Open http://localhost:8000 in your browser

# 3. Quick training
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3

# 4. Test results in chat
```

### Advanced Workflow
```bash
# 1. Check system
python dataset_command_interface.py status

# 2. Train on multiple datasets
python dataset_command_interface.py add-wikipedia --topics physics --samples 500 --train
python dataset_command_interface.py quick-start --dataset squad --samples 1000 --epochs 8 --augment

# 3. Train on personal data
python dataset_command_interface.py train-local --path ./documents/ --epochs 10 --augment

# 4. Full pipeline
python dataset_command_interface.py full-pipeline --source huggingface --dataset arxiv --augment --epochs 15
```

---

This reference covers all the essential commands for using the Gyroidic AI System. Keep this handy while working with the system!
