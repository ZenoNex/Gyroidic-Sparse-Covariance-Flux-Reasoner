# Gyroidic AI System - Complete User Manual

**Everything you need to know to use the Gyroidic AI System**

This manual covers all the ways you can interact with the AI system, from simple chat to advanced dataset training.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Web Chat Interface](#web-chat-interface)
3. [Dataset Training Commands](#dataset-training-commands)
4. [Wikipedia Learning System](#wikipedia-learning-system)
5. [Image Processing](#image-processing)
6. [Testing and Verification](#testing-and-verification)
7. [System Management](#system-management)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Features](#advanced-features)

---

## 🚀 Getting Started

### Installation
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Verify everything works
python test_fixes_verification.py

# 3. Start the system
python src/ui/diegetic_terminal.py
```

### First Steps
1. **Chat with the AI**: Open http://localhost:8000 after starting the terminal
2. **Train on data**: Use the dataset commands to teach the AI new topics
3. **Upload images**: Drag images into the chat to have the AI analyze them
4. **Check status**: Monitor system health and storage usage

---

## 💬 Web Chat Interface

### Starting the Chat
```bash
# Start the backend server
python src/ui/diegetic_terminal.py

# Open your browser to: http://localhost:8000
```

### Chat Features

#### Basic Conversation
- Type messages and get AI responses
- The AI remembers the conversation context
- Responses are coherent and human-like

#### Image Analysis
- Drag and drop images into the chat
- The AI will analyze and describe the image
- Supports JPG, PNG, BMP, and other common formats

#### System Information
- View current system status
- See memory usage and performance
- Monitor training progress

### Chat Commands
While chatting, you can use these special commands:

- `status` - Show system health
- `memory` - Display conversation memory
- `clear` - Clear conversation history
- `help` - Show available commands

---

## 🎯 Dataset Training Commands

### Main Command Interface
**File**: `dataset_command_interface.py`

This is your primary tool for training the AI on different datasets.

### Quick Start Training

#### Popular Datasets
```bash
# Movie reviews (teaches sentiment and opinions)
python dataset_command_interface.py quick-start --dataset imdb --samples 1000 --epochs 5

# Question-answering (teaches factual responses)
python dataset_command_interface.py quick-start --dataset squad --samples 500 --epochs 3

# Wikipedia articles (teaches general knowledge)
python dataset_command_interface.py quick-start --dataset wikitext --samples 2000 --epochs 10

# Scientific papers (teaches technical knowledge)
python dataset_command_interface.py quick-start --dataset arxiv --samples 800 --epochs 8

# Programming code (teaches coding)
python dataset_command_interface.py quick-start --dataset codeparrot --samples 1200 --epochs 6
```

#### Understanding the Options
- `--dataset`: Which dataset to use (see available list below)
- `--samples`: How many examples to learn from (more = better but slower)
- `--epochs`: How many times to go through the data (more = deeper learning)
- `--augment`: Use advanced data expansion techniques

### Wikipedia Learning
```bash
# Learn about physics
python dataset_command_interface.py add-wikipedia --topics physics --samples 500 --train

# Learn about mathematics
python dataset_command_interface.py add-wikipedia --topics mathematics --samples 300 --train

# Learn about computer science
python dataset_command_interface.py add-wikipedia --topics computer_science --samples 400 --train

# Learn about philosophy
python dataset_command_interface.py add-wikipedia --topics philosophy --samples 350 --train

# Learn about biology
python dataset_command_interface.py add-wikipedia --topics biology --samples 300 --train

# Custom topics (comma-separated)
python dataset_command_interface.py add-wikipedia --topics "Quantum_mechanics,Relativity,Thermodynamics" --samples 400 --train
```

### Training on Your Files
```bash
# Train on a folder of documents
python dataset_command_interface.py train-local --path ./documents/ --epochs 10 --augment

# Train on a specific file
python dataset_command_interface.py train-local --path ./my_book.txt --epochs 5

# Train on multiple file types
python dataset_command_interface.py train-local --path ./mixed_data/ --epochs 8 --augment
```

### Advanced Training Pipeline
```bash
# Full pipeline with HuggingFace dataset
python dataset_command_interface.py full-pipeline --source huggingface --dataset squad --augment --epochs 20

# Full pipeline with Wikipedia
python dataset_command_interface.py full-pipeline --source wikipedia --dataset "physics,mathematics" --augment --epochs 25

# Full pipeline with local files
python dataset_command_interface.py full-pipeline --source local --dataset ./my_data/ --augment --epochs 15
```

### System Management Commands
```bash
# List all available datasets
python dataset_command_interface.py list-datasets

# Check system status and storage usage
python dataset_command_interface.py status
```

---

## 📚 Wikipedia Learning System

### Web Interface
```bash
# Start the backend (includes Wikipedia trainer)
python src/ui/diegetic_backend.py

# Open: http://localhost:8000/wikipedia-trainer
```

### Wikipedia Trainer Features

#### Search and Download
- Search for Wikipedia articles by topic
- Preview articles before downloading
- Batch download multiple related articles

#### Content Processing
- Automatically clean Wikipedia formatting
- Extract key concepts and facts
- Remove citations while preserving mathematics
- Filter out navigation and metadata

#### Training Integration
- Train directly from the web interface
- Monitor training progress in real-time
- Adjust training parameters on the fly

### Command Line Wikipedia Training
```bash
# Predefined topic collections
python dataset_command_interface.py add-wikipedia --topics physics --samples 500 --train
python dataset_command_interface.py add-wikipedia --topics mathematics --samples 300 --train
python dataset_command_interface.py add-wikipedia --topics computer_science --samples 400 --train

# Custom article lists
python dataset_command_interface.py add-wikipedia --topics "Machine_learning,Neural_networks,Deep_learning" --samples 300 --train
```

---

## 🖼️ Image Processing

### Image Analysis in Chat
1. Start the chat interface: `python src/ui/diegetic_terminal.py`
2. Open http://localhost:8000
3. Drag and drop images into the chat
4. The AI will analyze and describe the images

### Image Processing Features
- **137-dimensional fingerprints**: Efficient image representation
- **Color analysis**: RGB and luminance histograms
- **Texture detection**: Surface patterns and variations
- **Edge detection**: Boundaries and shapes
- **Content description**: What the AI sees in the image

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

### Image Training Commands
```bash
# Test image processing
python test_image_simple.py

# Test full image integration
python test_image_integration.py

# Create test images and analyze them
python image_extension.py
```

---

## 🧪 Testing and Verification

### Core System Tests
```bash
# Test that bug fixes are working
python test_fixes_verification.py

# Test image processing without dependencies
python test_image_simple.py

# Test full image integration
python test_image_integration.py
```

### Feature Tests
```bash
# Test Wikipedia integration
python test_enhanced_wikipedia_system.py

# Test association learning
python test_enhanced_association_learning.py

# Test dataset augmentation
python test_mandelbulb_augmentation.py
```

### System Phase Tests
```bash
# Test Phase 3 features (dyad system)
python test_phase3_dyad_system.py

# Test Phase 4 features (advanced analysis)
python test_phase4_advanced_features.py

# Test repair integration
python test_repair_integration.py
```

### Quick System Check
```bash
# Run the most important tests
python test_fixes_verification.py
python test_enhanced_wikipedia_system.py
python dataset_command_interface.py status
```

---

## ⚙️ System Management

### Checking System Status
```bash
# Comprehensive status check
python dataset_command_interface.py status
```

This shows:
- Storage usage and available space
- Loaded datasets and their sizes
- Active models and their parameters
- System health indicators
- PyTorch and CUDA status

### Managing Storage
```bash
# Check what's using space
python dataset_command_interface.py status

# List all datasets
python dataset_command_interface.py list-datasets

# Clean up old data (manual - delete files in datasets/ folder)
```

### Starting and Stopping Services

#### Web Chat Interface
```bash
# Start
python src/ui/diegetic_terminal.py

# Stop: Ctrl+C in the terminal
```

#### Backend Server
```bash
# Start
python src/ui/diegetic_backend.py

# Stop: Ctrl+C in the terminal
```

#### Wikipedia Trainer
```bash
# Start backend first
python src/ui/diegetic_backend.py

# Then open: http://localhost:8000/wikipedia-trainer
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### "PIL.Image has no attribute 'Resampling'"
**Problem**: Older version of Pillow library  
**Solution**:
```bash
pip install --upgrade Pillow
```

#### "'float' object has no attribute 'numel'"
**Problem**: Type handling bug (should be fixed)  
**Solution**:
```bash
python test_fixes_verification.py
```

#### "The size of tensor a (768) must match the size of tensor b (256)"
**Problem**: Dimension mismatch between image and text embeddings  
**Solution**: This is fixed in the latest version. The system uses learnable projection to align dimensions while preserving learned representations.
```bash
python test_image_simple.py
```
For technical details, see [Tensor Dimension Fix Guide](TENSOR_DIMENSION_FIX.md).

#### "Backend Connectivity Lost"
**Problem**: Web interface can't connect to backend  
**Solutions**:
1. Make sure backend is running: `python src/ui/diegetic_backend.py`
2. Check Windows Firewall (allow Python through port 8000)
3. Try restarting both frontend and backend

#### "Out of storage space"
**Problem**: Datasets are using too much space  
**Solutions**:
```bash
# Check usage
python dataset_command_interface.py status

# Use smaller datasets
python dataset_command_interface.py quick-start --dataset imdb --samples 500

# Clean up old datasets (manually delete files in datasets/ folder)
```

#### "Training too slow"
**Problem**: Training is taking too long  
**Solutions**:
```bash
# Use fewer samples and epochs
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3

# Use smaller batch sizes (add --batch-size 2 to commands)
```

#### "Dataset not found"
**Problem**: Typo in dataset name  
**Solution**:
```bash
# Check available datasets
python dataset_command_interface.py list-datasets
```

#### "Connection timeout" or "Download failed"
**Problem**: Network issues downloading datasets  
**Solutions**:
1. Check internet connection
2. Try again later
3. Use local files instead: `train-local --path ./my_files/`

### Getting Help

1. **Check this troubleshooting section first**
2. **Run verification tests**: `python test_fixes_verification.py`
3. **Check system status**: `python dataset_command_interface.py status`
4. **Review error messages carefully** - they often contain the solution
5. **Try with smaller datasets first** to isolate the problem

### Performance Tips

#### For Faster Training
- Use fewer samples (500-1000 instead of 2000+)
- Use fewer epochs (3-5 instead of 10+)
- Skip augmentation initially
- Use smaller models

#### For Better Results
- Use more samples (2000+ if storage allows)
- Train for more epochs (8-12)
- Use augmentation (`--augment`)
- Combine multiple related datasets

#### For Storage Efficiency
- Monitor usage regularly: `python dataset_command_interface.py status`
- Use appropriate sample sizes for your storage
- Clean up old datasets you no longer need
- Use compression-friendly formats

---

## 🔬 Advanced Features

<details>
<summary>Click to expand advanced technical features</summary>

### Advanced Dataset Configuration

#### Model Architecture Options
```bash
--functionals N     # Number of reasoning components (3-8, default: 5)
--hidden-dim N      # Model size (256, 512, 768, default: 256)
--poly-degree N     # Mathematical complexity (3-7, default: 4)
--num-heads N       # Attention mechanisms (8, 12, 16, default: 8)
```

#### Training Parameters
```bash
--batch-size N      # Training batch size (2-8, default: 4)
--learning-rate F   # Learning speed (1e-5 to 1e-3, default: 1e-4)
--evolution-rate F  # Evolution speed (0.01-0.05, default: 0.02)
--checkpoint        # Save training progress
--checkpoint-interval N  # Save every N epochs
```

#### Data Augmentation
The system uses "Mandelbulb-Gyroidic Augmentation":
- Expands small datasets using mathematical principles
- Preserves data structure and meaning
- Uses fractal and topological techniques
- Maintains training efficiency

### Expert-Level Commands

#### Custom Model Creation
```bash
# Create a specialized model
python dataset_ingestion_system.py create-model \
  --name "expert_model" \
  --type temporal \
  --functionals 8 \
  --poly-degree 6 \
  --hidden-dim 768 \
  --num-heads 16
```

#### Advanced Training Setup
```bash
# Setup training with all options
python dataset_ingestion_system.py setup-training \
  --model "expert_model" \
  --dataset "my_dataset" \
  --epochs 25 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --evolution-rate 0.01 \
  --mandelbulb \
  --augmentation-factor 4 \
  --checkpoint \
  --checkpoint-interval 3
```

### Research Features

#### Anti-Lobotomy Architecture
- Prevents AI from becoming oversimplified
- Maintains complexity and nuanced reasoning
- Uses evolutionary selection instead of gradient descent
- Preserves structural integrity during training

#### Topological Reasoning
- Uses gyroidal minimal surfaces for coherence
- Maintains topological properties during learning
- Detects and repairs structural violations
- Ensures mathematical consistency

#### Temporal Association Learning
- Builds long-term memory through dyadic relationships
- Creates persistent associations between concepts
- Enables contextual understanding across conversations
- Supports incremental learning from interactions

</details>

---

## 📊 Understanding the System

### What Makes This AI Different

#### Coherent Responses
Unlike many AI systems that produce garbled or inconsistent output, this system generates coherent, human-like responses through breakthrough architectural innovations.

#### True Learning
The system can genuinely learn from new data, not just retrieve pre-trained information. It forms new associations and adapts its responses based on what it learns.

#### Storage Efficient
Designed to work within practical storage limits (100GB), making it accessible for personal and research use without requiring massive computational resources.

#### Anti-Lobotomy Design
Prevents the common AI problem of "lobotomy" - where systems become oversimplified and lose their ability to handle complex reasoning.

### How Training Works

1. **Data Ingestion**: The system processes your chosen dataset
2. **Fingerprint Creation**: Creates compact representations of the data
3. **Association Learning**: Builds connections between concepts
4. **Memory Formation**: Stores learned patterns for future use
5. **Response Generation**: Uses learned patterns to generate coherent responses

### Storage and Performance

#### Storage Breakdown (per 1000 samples)
- Raw text data: ~50-100 MB
- Processed fingerprints: ~5 MB
- Embeddings: ~15 MB
- Augmented data: ~20-40 MB
- **Total**: ~90-160 MB

#### Performance Expectations
- **Small datasets (500 samples)**: 5-10 minutes training
- **Medium datasets (2000 samples)**: 20-40 minutes training
- **Large datasets (5000+ samples)**: 1-3 hours training

---

This manual covers all the ways to interact with the Gyroidic AI System. Start with the basic chat interface and gradually explore the more advanced training features as you become comfortable with the system.

For technical details and research information, see the documentation in the `docs/` folder.