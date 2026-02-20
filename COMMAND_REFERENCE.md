# Gyroidic AI System - Command Reference

**Quick reference for all command-line interfaces**

---

## 🚀 Essential Commands

### Start the System
```bash
# Web chat interface
python src/ui/diegetic_terminal.py
# Then open: http://localhost:8000

# Backend server (for Wikipedia trainer)
python src/ui/diegetic_backend.py
# Then open: http://localhost:8000/wikipedia-trainer
```

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

## 📚 Dataset Training Commands

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

**Sources:** `huggingface`, `wikipedia`, `local`

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

## 🌐 Web Interfaces

### Chat Interface
```bash
# Start server
python src/ui/diegetic_terminal.py

# Open browser to: http://localhost:8000
```

**Features:**
- Chat with AI
- Upload and analyze images
- View system status
- Real-time responses

### Wikipedia Trainer
```bash
# Start backend
python src/ui/diegetic_backend.py

# Open browser to: http://localhost:8000/wikipedia-trainer
```

**Features:**
- Search Wikipedia articles
- Download and process content
- Train on specific topics
- Monitor progress

---

## 🧪 Testing Commands

### Core Tests
```bash
# Test bug fixes (run this first)
python test_fixes_verification.py

# Test image processing
python test_image_simple.py
python test_image_integration.py

# Test Wikipedia system
python test_enhanced_wikipedia_system.py
```

### Feature Tests
```bash
# Test association learning
python test_enhanced_association_learning.py

# Test dataset augmentation
python test_mandelbulb_augmentation.py

# Test repair systems
python test_repair_integration.py
```

### System Phase Tests
```bash
# Test Phase 3 (dyad system)
python test_phase3_dyad_system.py

# Test Phase 4 (advanced features)
python test_phase4_advanced_features.py

# Test enhanced fingerprints
python test_enhanced_fingerprint.py
```

---

## ⚙️ Configuration Options

### Common Options
```bash
--samples N          # Number of examples (500-5000)
--epochs N           # Training rounds (3-20)
--augment           # Use data expansion
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

## 📊 Quick Reference Tables

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

## 🚨 Troubleshooting Quick Fixes

### Common Errors
```bash
# PIL version error
pip install --upgrade Pillow

# Type handling error (should be fixed)
python test_fixes_verification.py

# Backend connection lost
python src/ui/diegetic_backend.py

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

## 📁 File Locations

### Main Scripts
- `dataset_command_interface.py` - Main dataset training
- `src/ui/diegetic_terminal.py` - Web chat interface
- `src/ui/diegetic_backend.py` - Backend server
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

## 🎯 Workflow Examples

### Beginner Workflow
```bash
# 1. Test system
python test_fixes_verification.py

# 2. Start chat
python src/ui/diegetic_terminal.py
# Open http://localhost:8000

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