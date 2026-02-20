# Complete Guide to Dataset Training

**Learn how to train the Gyroidic AI System on any dataset**

This guide shows you how to train the AI system on different types of data, from movie reviews to Wikipedia articles to your own documents.

---

## 🚀 Quick Start Examples

### Train on Movie Reviews (5 minutes)
```bash
# This will download movie reviews and train the AI to understand sentiment
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3
```

### Learn Physics from Wikipedia (10 minutes)
```bash
# This downloads physics articles and teaches the AI about physics
python dataset_command_interface.py add-wikipedia --topics physics --samples 300 --train
```

### Train on Your Documents (varies)
```bash
# This trains the AI on your own files
python dataset_command_interface.py train-local --path ./my_documents/ --epochs 5
```

---

## 📚 Understanding Datasets

### What is a Dataset?
A dataset is a collection of examples that the AI learns from. Think of it like a textbook - the more examples the AI sees, the better it gets at understanding and generating similar content.

### Types of Datasets

#### Text Datasets
- **Movie Reviews (IMDB)**: Teaches the AI about opinions and sentiment
- **Question-Answer Pairs (Squad)**: Teaches the AI to answer questions
- **Wikipedia Articles**: Teaches the AI factual knowledge
- **Code Repositories**: Teaches the AI about programming
- **Scientific Papers**: Teaches the AI about research and technical topics

#### Image Datasets
- **Photo Captions (COCO)**: Teaches the AI to describe images
- **Flickr Photos**: More image-text associations

#### Your Own Data
- **Documents**: PDFs, Word files, text files
- **Books**: Any text content you want the AI to learn from
- **Conversations**: Chat logs or transcripts

---

## 🎯 Command Reference

### Main Command: `dataset_command_interface.py`

This is your main tool for training the AI. All commands follow this pattern:
```bash
python dataset_command_interface.py [command] [options]
```

### Available Commands

#### `quick-start` - Easiest Way to Begin
Automatically downloads a popular dataset and starts training.

```bash
python dataset_command_interface.py quick-start --dataset [name] --samples [number] --epochs [number]
```

**Options:**
- `--dataset`: Which dataset to use (see list below)
- `--samples`: How many examples to use (more = better but slower)
- `--epochs`: How many times to go through the data (more = better but slower)
- `--augment`: Use advanced data expansion (optional)

**Examples:**
```bash
# Small, fast training (good for testing)
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3

# Medium training (good balance)
python dataset_command_interface.py quick-start --dataset squad --samples 1000 --epochs 5

# Large training (best results but slower)
python dataset_command_interface.py quick-start --dataset wikitext --samples 2000 --epochs 10
```

#### `add-wikipedia` - Learn from Wikipedia
Downloads and trains on Wikipedia articles about specific topics.

```bash
python dataset_command_interface.py add-wikipedia --topics [topic] --samples [number] --train
```

**Options:**
- `--topics`: What to learn about (see topic list below)
- `--samples`: How many articles per topic
- `--train`: Start training immediately after downloading

**Examples:**
```bash
# Learn physics
python dataset_command_interface.py add-wikipedia --topics physics --samples 500 --train

# Learn multiple topics
python dataset_command_interface.py add-wikipedia --topics "Quantum_mechanics,Relativity" --samples 300 --train

# Just download without training
python dataset_command_interface.py add-wikipedia --topics mathematics --samples 400
```

#### `train-local` - Use Your Own Files
Trains the AI on your own documents and files.

```bash
python dataset_command_interface.py train-local --path [folder/file] --epochs [number]
```

**Options:**
- `--path`: Path to your files or folder
- `--epochs`: How many training rounds
- `--samples`: Limit number of files (optional)
- `--augment`: Use data expansion (optional)

**Examples:**
```bash
# Train on a folder of documents
python dataset_command_interface.py train-local --path ./documents/ --epochs 10

# Train on a single file
python dataset_command_interface.py train-local --path ./my_book.txt --epochs 5

# Train with data expansion
python dataset_command_interface.py train-local --path ./data/ --epochs 8 --augment
```

#### `full-pipeline` - Advanced Training
Uses all advanced features for the best results.

```bash
python dataset_command_interface.py full-pipeline --source [type] --dataset [name] --augment --epochs [number]
```

**Options:**
- `--source`: Where the data comes from (huggingface, wikipedia, local)
- `--dataset`: Dataset name or path
- `--augment`: Use advanced data expansion
- `--epochs`: Training rounds
- `--samples`: Limit data size

**Examples:**
```bash
# Advanced HuggingFace training
python dataset_command_interface.py full-pipeline --source huggingface --dataset squad --augment --epochs 20

# Advanced Wikipedia training
python dataset_command_interface.py full-pipeline --source wikipedia --dataset "physics,math" --augment --epochs 25

# Advanced local file training
python dataset_command_interface.py full-pipeline --source local --dataset ./my_data/ --augment --epochs 15
```

#### `list-datasets` - See What's Available
Shows all available datasets and their details.

```bash
python dataset_command_interface.py list-datasets
```

#### `status` - Check System Health
Shows storage usage, loaded datasets, and system status.

```bash
python dataset_command_interface.py status
```

---

## 📖 Available Datasets

### Popular Text Datasets

| Name | Description | Good For | Size (1k samples) |
|------|-------------|----------|-------------------|
| `imdb` | Movie reviews | Learning opinions, sentiment | ~100 MB |
| `squad` | Question-answer pairs | Teaching Q&A skills | ~120 MB |
| `wikitext` | Wikipedia articles | General knowledge | ~90 MB |
| `openwebtext` | Web content | Diverse writing styles | ~150 MB |
| `codeparrot` | Programming code | Code understanding | ~110 MB |
| `github_code` | Code repositories | Software development | ~130 MB |
| `arxiv` | Scientific papers | Research, technical topics | ~140 MB |
| `pubmed` | Medical research | Healthcare, biology | ~120 MB |

### Wikipedia Topic Collections

| Topic | Articles Included | Good For |
|-------|------------------|----------|
| `physics` | Quantum mechanics, relativity, thermodynamics, particle physics | Science education |
| `mathematics` | Linear algebra, calculus, topology, number theory | Math tutoring |
| `computer_science` | Machine learning, algorithms, data structures | Programming help |
| `philosophy` | Philosophy of mind, logic, ethics | Critical thinking |
| `biology` | Molecular biology, evolution, genetics | Life sciences |

### Multimodal Datasets

| Name | Description | Features |
|------|-------------|----------|
| `coco` | Images with captions | Image understanding |
| `flickr30k` | Photo descriptions | Visual-text connections |

---

## ⚙️ Configuration Guide

### Understanding the Options

#### `--samples` (Number of Examples)
- **Small (500-1000)**: Fast training, good for testing
- **Medium (1000-3000)**: Good balance of speed and quality
- **Large (3000+)**: Best results but slower

#### `--epochs` (Training Rounds)
- **Few (3-5)**: Quick training, basic learning
- **Medium (5-10)**: Good learning, reasonable time
- **Many (10+)**: Deep learning, takes longer

#### `--augment` (Data Expansion)
- Automatically creates more training examples from your data
- Uses mathematical techniques to expand small datasets
- Recommended for datasets under 1000 samples

### Storage Planning

The system is designed to work within 100GB of storage. Here's how to plan:

#### Storage per Dataset (approximate)
- **Text + Processing**: ~90-160 MB per 1000 samples
- **Images + Processing**: ~200-400 MB per 1000 samples
- **Augmented Data**: +50% storage usage

#### Recommended Limits
- **Small datasets**: 500-1000 samples
- **Medium datasets**: 1000-3000 samples  
- **Large datasets**: 3000-5000 samples

#### Check Your Usage
```bash
python dataset_command_interface.py status
```

---

## 🔄 Training Workflow

### Step 1: Choose Your Data
Decide what you want the AI to learn:
- **General knowledge**: Use Wikipedia topics
- **Specific skills**: Use targeted datasets (like code or Q&A)
- **Personal assistant**: Use your own documents

### Step 2: Start Small
Begin with a small dataset to test:
```bash
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3
```

### Step 3: Check Results
After training, test the AI:
```bash
python src/ui/diegetic_terminal.py
# Open http://localhost:8000 and chat with it
```

### Step 4: Scale Up
If results are good, train on more data:
```bash
python dataset_command_interface.py quick-start --dataset imdb --samples 2000 --epochs 8 --augment
```

### Step 5: Add More Topics
Expand the AI's knowledge:
```bash
python dataset_command_interface.py add-wikipedia --topics physics --samples 500 --train
python dataset_command_interface.py add-wikipedia --topics mathematics --samples 300 --train
```

---

## 🚨 Troubleshooting

### Common Issues

#### "Dataset not found"
Make sure you're using the exact dataset name from the list above.
```bash
# Check available datasets
python dataset_command_interface.py list-datasets
```

#### "Out of storage space"
Check your usage and use smaller datasets:
```bash
python dataset_command_interface.py status
# Use fewer samples
python dataset_command_interface.py quick-start --dataset imdb --samples 500
```

#### "Training too slow"
Reduce samples and epochs:
```bash
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3
```

#### "Connection failed" or "Backend error"
Make sure the backend is running:
```bash
python src/ui/diegetic_backend.py
```

### Getting Better Results

#### For Better Text Generation
1. Use larger datasets (2000+ samples)
2. Train for more epochs (8-12)
3. Use augmentation (`--augment`)
4. Combine multiple datasets

#### For Faster Training
1. Use fewer samples (500-1000)
2. Use fewer epochs (3-5)
3. Skip augmentation initially
4. Use smaller models

#### For Specific Domains
1. Use domain-specific datasets (arxiv for science, codeparrot for programming)
2. Add relevant Wikipedia topics
3. Include your own domain documents
4. Use full pipeline for best results

---

## 📊 Monitoring Training

### Check System Status
```bash
python dataset_command_interface.py status
```

This shows:
- Storage usage
- Loaded datasets
- Available models
- System health

### Watch Training Progress
Training will show progress like:
```
Epoch 1/5: Loss = 0.234
Epoch 2/5: Loss = 0.198
Epoch 3/5: Loss = 0.167
...
```

Lower loss numbers mean better learning.

### Test Your AI
After training, test it:
```bash
python src/ui/diegetic_terminal.py
# Open http://localhost:8000
# Type messages and see responses
```

---

## 🎯 Best Practices

### For Beginners
1. Start with `quick-start` and small datasets
2. Use 500 samples and 3 epochs initially
3. Test results before scaling up
4. Check storage usage regularly

### For Better Results
1. Use multiple related datasets
2. Include Wikipedia knowledge
3. Use augmentation for small datasets
4. Train for more epochs (8-12)

### For Efficiency
1. Monitor storage usage
2. Use appropriate sample sizes
3. Don't over-train (diminishing returns after 15 epochs)
4. Clean up old datasets you don't need

---

## 🔬 Advanced Features

<details>
<summary>Click for advanced technical options</summary>

### Advanced Configuration

#### Model Architecture Options
```bash
--functionals N     # Number of reasoning components (3-8)
--hidden-dim N      # Model size (256, 512, 768)
--poly-degree N     # Mathematical complexity (3-7)
--num-heads N       # Attention mechanisms (8, 12, 16)
```

#### Training Parameters
```bash
--batch-size N      # Training batch size (2-8)
--learning-rate F   # Learning speed (1e-5 to 1e-3)
--evolution-rate F  # Evolution speed (0.01-0.05)
--checkpoint        # Save training progress
```

#### Data Augmentation
The system uses "Mandelbulb-Gyroidic Augmentation" - a mathematical technique that:
- Expands small datasets intelligently
- Preserves data structure and meaning
- Uses fractal and topological principles
- Maintains training efficiency

</details>

---

This guide covers everything you need to train the Gyroidic AI System on any dataset. Start with the quick examples and gradually explore more advanced features as you become comfortable with the system.

python dataset_ingestion_system.py setup-training \
  --model "document_reasoner" \
  --dataset "my_documents" \
  --epochs 8 \
  --mandelbulb \
  --augmentation-factor 2

python dataset_ingestion_system.py train \
  --model "document_reasoner" \
  --dataset "my_documents"
```

---

## 📊 Dataset Sources

### HuggingFace Hub
Access thousands of datasets from the HuggingFace ecosystem:

```bash
# Popular text datasets
python dataset_ingestion_system.py add-dataset --name "squad" --source huggingface --path "squad" --preprocessing text
python dataset_ingestion_system.py add-dataset --name "wikitext" --source huggingface --path "wikitext" --preprocessing text
python dataset_ingestion_system.py add-dataset --name "openwebtext" --source huggingface --path "openwebtext" --preprocessing text --max-samples 5000

# Conversation datasets
python dataset_ingestion_system.py add-dataset --name "persona_chat" --source huggingface --path "persona_chat" --preprocessing text
python dataset_ingestion_system.py add-dataset --name "empathetic_dialogues" --source huggingface --path "empathetic_dialogues" --preprocessing text

# Code datasets
python dataset_ingestion_system.py add-dataset --name "code_search" --source huggingface --path "code_search_net" --preprocessing text
```

### Kaggle Datasets
Access Kaggle's vast dataset collection:

```bash
# Requires Kaggle API setup: pip install kaggle
# Configure with your API key from https://www.kaggle.com/account

# Text datasets
python dataset_ingestion_system.py add-dataset --name "news_articles" --source kaggle --path "rmisra/news-category-dataset" --preprocessing text
python dataset_ingestion_system.py add-dataset --name "amazon_reviews" --source kaggle --path "bittlingmayer/amazonreviews" --preprocessing text --max-samples 10000

# Scientific datasets
python dataset_ingestion_system.py add-dataset --name "arxiv_papers" --source kaggle --path "Cornell-University/arxiv" --preprocessing text --max-samples 1000
```

### Wikipedia Articles
Curated knowledge extraction:

```bash
# Single topic
python dataset_ingestion_system.py add-dataset --name "ai_articles" --source wikipedia --path "Artificial_intelligence" --preprocessing text

# Multiple topics
python dataset_ingestion_system.py add-dataset --name "science_topics" --source wikipedia --path "Physics,Chemistry,Biology,Mathematics" --preprocessing text

# From file (one topic per line)
echo -e "Machine_learning\nDeep_learning\nNeural_network\nArtificial_intelligence" > topics.txt
python dataset_ingestion_system.py add-dataset --name "ml_knowledge" --source wikipedia --path "topics.txt" --preprocessing text
```

### Local Files
Process your own data:

```bash
# Text files
python dataset_ingestion_system.py add-dataset --name "my_texts" --source local --path "./text_data/" --preprocessing text

# JSON/JSONL files
python dataset_ingestion_system.py add-dataset --name "structured_data" --source local --path "./data.jsonl" --preprocessing text

# CSV files
python dataset_ingestion_system.py add-dataset --name "tabular_data" --source local --path "./data.csv" --preprocessing tabular
```

### URL Downloads
Direct dataset downloads:

```bash
# Download and process
python dataset_ingestion_system.py add-dataset --name "web_dataset" --source url --path "https://example.com/dataset.zip" --preprocessing text
```

---

## 🏗️ Model Configuration

### Temporal Models
The primary model type for sequential reasoning:

```bash
# Basic temporal model
python dataset_ingestion_system.py create-model \
  --name "basic_temporal" \
  --type temporal \
  --functionals 5 \
  --hidden-dim 256

# Advanced temporal model
python dataset_ingestion_system.py create-model \
  --name "advanced_temporal" \
  --type temporal \
  --functionals 8 \
  --poly-degree 6 \
  --hidden-dim 512 \
  --input-dim 768

# Specialized models for different domains
python dataset_ingestion_system.py create-model \
  --name "code_reasoner" \
  --type temporal \
  --functionals 6 \
  --poly-degree 4 \
  --hidden-dim 384

python dataset_ingestion_system.py create-model \
  --name "science_reasoner" \
  --type temporal \
  --functionals 10 \
  --poly-degree 7 \
  --hidden-dim 768
```

### Model Parameters

- **`--functionals`**: Number of polynomial co-prime functionals (K)
  - More functionals = higher capacity but slower training
  - Recommended: 5-8 for most tasks, 10+ for complex domains

- **`--poly-degree`**: Degree of polynomial basis functions
  - Higher degree = more expressive but risk of overfitting
  - Recommended: 4-6 for most tasks

- **`--hidden-dim`**: Hidden layer dimension
  - Larger = more capacity but more parameters
  - Recommended: 256-512 for most tasks

- **`--input-dim`**: Input embedding dimension
  - Should match your embedding system (768 for many transformers)

---

## 🎯 Training Configuration

### Basic Training Setup

```bash
python dataset_ingestion_system.py setup-training \
  --model "my_model" \
  --dataset "my_dataset" \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --evolution-rate 0.02
```

### Advanced Training with Mandelbulb Augmentation

```bash
python dataset_ingestion_system.py setup-training \
  --model "advanced_model" \
  --dataset "complex_dataset" \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --evolution-rate 0.01 \
  --mandelbulb \
  --augmentation-factor 3
```

### Training Parameters

- **`--epochs`**: Number of training epochs
  - More epochs = better learning but longer training
  - Recommended: 10-20 for most datasets

- **`--batch-size`**: Training batch size
  - Larger batches = more stable gradients but more memory
  - Recommended: 4-8 for most systems

- **`--learning-rate`**: Neural network learning rate
  - Standard Adam learning rate for neural components
  - Recommended: 1e-4 to 1e-5

- **`--evolution-rate`**: Trust scalar evolution rate
  - Rate of evolutionary trust selection (not gradient descent)
  - Recommended: 0.01-0.02

- **`--mandelbulb`**: Enable Mandelbulb-Gyroidic augmentation
  - Geometric dataset expansion using fractal principles
  - Increases training data while preserving topological structure

- **`--augmentation-factor`**: Mandelbulb augmentation multiplier
  - How many augmented samples per original sample
  - Recommended: 2-3 (higher values increase training time)

---

## 🌀 Mandelbulb-Gyroidic Augmentation

The system includes a novel geometric augmentation method that combines:
- **Mandelbulb fractals** for self-similar recursive structures
- **Gyroidic minimal surfaces** for energy-minimizing constraints
- **Sparse covariance preservation** for maintaining data relationships

### When to Use Mandelbulb Augmentation

✅ **Recommended for:**
- Small datasets that need expansion
- Complex domains requiring topological coherence
- Training that benefits from geometric variations
- Systems needing robust generalization

❌ **Not recommended for:**
- Very large datasets (>100k samples)
- Simple classification tasks
- Time-critical training scenarios
- Memory-constrained environments

### Augmentation Configuration

The system automatically configures augmentation parameters, but you can understand what's happening:

```python
# Internal configuration (automatic)
AugmentationConfig(
    mandelbulb_power=8,        # Fractal complexity
    max_iterations=50,         # Convergence iterations
    gyroid_tolerance=1e-3,     # Surface projection accuracy
    sparsity_threshold=0.1,    # Covariance preservation
    pressure_adaptation=True   # Dynamic parameter adjustment
)
```

---

## 📈 Monitoring and Management

### List System State

```bash
# List all datasets
python dataset_ingestion_system.py list-datasets

# List all models
python dataset_ingestion_system.py list-models

# List training sessions
python dataset_ingestion_system.py list-training

# List everything
python dataset_ingestion_system.py list-all
```

### Example Output

```
📊 Available Datasets:
   • imdb_reviews
     Source: huggingface - imdb
     Preprocessing: text
     Samples: 1000
     Augmentation: True

🏗️ Available Models:
   • text_reasoner
     Parameters: 1,234,567
     Functionals: 5
     Trust mean: 0.856
     Fossilized: 2/5

🎯 Training Sessions:
   • text_reasoner_imdb_reviews
     Status: Complete
     Epochs: 10/10
     Runtime: 1234.5s
```

---

## 🔍 Anti-Lobotomy Compliance

The system automatically enforces anti-lobotomy principles:

### ✅ Compliance Checks

1. **No Hardcoded Primes**: Uses polynomial co-prime functionals
2. **Evolutionary Trust**: Trust scalars evolve via selection, not gradients
3. **Structural Honesty**: No placeholder implementations
4. **Non-Teleological Flow**: Survivorship pressure, not loss minimization
5. **Mathematical Integrity**: Proper polynomial basis functions

### 🚨 Violation Detection

The system automatically detects and prevents:
- Hardcoded prime sequences `[2, 3, 5, 7, 11, ...]`
- Placeholder implementations `torch.randn()` for mathematical systems
- Gradient descent on trust scalars (teleological violation)
- Missing polynomial configurations
- Invalid evolutionary components

---

## 🎓 Advanced Usage Patterns

### 1. Multi-Domain Knowledge Integration

```bash
# Create specialized models for different domains
python dataset_ingestion_system.py create-model --name "science_model" --functionals 8 --hidden-dim 512
python dataset_ingestion_system.py create-model --name "literature_model" --functionals 6 --hidden-dim 384
python dataset_ingestion_system.py create-model --name "code_model" --functionals 7 --hidden-dim 448

# Add domain-specific datasets
python dataset_ingestion_system.py add-dataset --name "physics" --source wikipedia --path "Physics,Quantum_mechanics,Relativity"
python dataset_ingestion_system.py add-dataset --name "literature" --source huggingface --path "bookcorpus" --max-samples 5000
python dataset_ingestion_system.py add-dataset --name "code" --source huggingface --path "code_search_net" --max-samples 3000

# Train each model on its domain
for model in science_model literature_model code_model; do
  for dataset in physics literature code; do
    if [[ "$model" == *"${dataset%s}"* ]]; then
      python dataset_ingestion_system.py setup-training --model "$model" --dataset "$dataset" --epochs 15 --mandelbulb
      python dataset_ingestion_system.py train --model "$model" --dataset "$dataset"
    fi
  done
done
```

### 2. Progressive Training Pipeline

```bash
# Stage 1: Basic training
python dataset_ingestion_system.py setup-training --model "progressive_model" --dataset "basic_data" --epochs 5
python dataset_ingestion_system.py train --model "progressive_model" --dataset "basic_data"

# Stage 2: Advanced training with augmentation
python dataset_ingestion_system.py setup-training --model "progressive_model" --dataset "advanced_data" --epochs 10 --mandelbulb --augmentation-factor 2
python dataset_ingestion_system.py train --model "progressive_model" --dataset "advanced_data"

# Stage 3: Fine-tuning with complex data
python dataset_ingestion_system.py setup-training --model "progressive_model" --dataset "complex_data" --epochs 15 --mandelbulb --augmentation-factor 3 --evolution-rate 0.005
python dataset_ingestion_system.py train --model "progressive_model" --dataset "complex_data"
```

### 3. Comparative Model Training

```bash
# Create multiple model variants
python dataset_ingestion_system.py create-model --name "model_5f" --functionals 5 --hidden-dim 256
python dataset_ingestion_system.py create-model --name "model_8f" --functionals 8 --hidden-dim 256
python dataset_ingestion_system.py create-model --name "model_5f_large" --functionals 5 --hidden-dim 512

# Train all on same dataset
for model in model_5f model_8f model_5f_large; do
  python dataset_ingestion_system.py setup-training --model "$model" --dataset "comparison_data" --epochs 10 --mandelbulb
  python dataset_ingestion_system.py train --model "$model" --dataset "comparison_data"
done

# Compare results
python dataset_ingestion_system.py list-models
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Dataset Loading Fails**
```bash
# Check dataset path and permissions
ls -la /path/to/dataset
# Verify preprocessing type matches data
python dataset_ingestion_system.py add-dataset --name "test" --source local --path "./sample.txt" --preprocessing text
```

**2. Model Creation Fails**
```bash
# Check device availability
python -c "import torch; print(torch.cuda.is_available())"
# Reduce model size if memory issues
python dataset_ingestion_system.py create-model --name "small_model" --functionals 3 --hidden-dim 128
```

**3. Training Fails**
```bash
# Check model and dataset exist
python dataset_ingestion_system.py list-all
# Reduce batch size if memory issues
python dataset_ingestion_system.py setup-training --model "my_model" --dataset "my_data" --batch-size 2
```

**4. Anti-Lobotomy Violations**
```bash
# System automatically prevents violations, but if you see errors:
# - Check that you're using the official system (no manual modifications)
# - Verify all dependencies are properly installed
# - Report any violations as they indicate system corruption
```

### Performance Optimization

**Memory Usage:**
- Reduce `--batch-size` for memory-constrained systems
- Use `--max-samples` to limit dataset size
- Reduce `--hidden-dim` for smaller models

**Training Speed:**
- Disable `--mandelbulb` for faster training
- Reduce `--augmentation-factor`
- Use fewer `--epochs` for initial testing
- Reduce `--functionals` for simpler models

**Quality vs Speed:**
- More `--functionals` = better quality, slower training
- Higher `--poly-degree` = more expressive, risk of overfitting
- `--mandelbulb` = better generalization, longer training

---

## 📚 Dataset Recommendations

### Text Understanding
- **HuggingFace**: `imdb`, `squad`, `wikitext`, `openwebtext`
- **Wikipedia**: Core topics in your domain of interest
- **Local**: Your own text corpus, documentation, books

### Conversational AI
- **HuggingFace**: `persona_chat`, `empathetic_dialogues`, `blended_skill_talk`
- **Wikipedia**: Conversational topics, social sciences

### Code Understanding
- **HuggingFace**: `code_search_net`, `github_code`
- **Kaggle**: Programming contest datasets
- **Local**: Your codebase, documentation

### Scientific Reasoning
- **Wikipedia**: Scientific topics, mathematical concepts
- **Kaggle**: Scientific datasets, research papers
- **ArXiv**: Research paper abstracts and content

### Creative Writing
- **HuggingFace**: `bookcorpus`, `writingprompts`
- **Local**: Literature, creative writing samples
- **Wikipedia**: Literary topics, creative writing techniques

---

## 🎯 Best Practices

### Dataset Preparation
1. **Start Small**: Use `--max-samples 1000` for initial testing
2. **Clean Data**: Ensure text is properly encoded and formatted
3. **Domain Focus**: Use domain-specific datasets for specialized models
4. **Progressive Scaling**: Start with small datasets, then scale up

### Model Configuration
1. **Conservative Start**: Begin with 5 functionals, degree 4
2. **Scale Gradually**: Increase complexity based on results
3. **Monitor Trust**: Watch trust scalar evolution during training
4. **Fossilization**: Allow natural fossilization, don't force it

### Training Strategy
1. **Baseline First**: Train without Mandelbulb augmentation initially
2. **Add Augmentation**: Use Mandelbulb for improved generalization
3. **Monitor Metrics**: Watch survivorship pressure and coherence
4. **Save Checkpoints**: Enable checkpointing for long training runs

### System Maintenance
1. **Regular Monitoring**: Check anti-lobotomy compliance
2. **Clean Datasets**: Remove corrupted or low-quality data
3. **Model Comparison**: Train multiple variants for comparison
4. **Documentation**: Keep track of successful configurations

---

## 🚀 Getting Started Checklist

- [ ] Install dependencies: `torch`, `numpy`, `requests`
- [ ] Optional: Install `datasets` for HuggingFace support
- [ ] Optional: Install `kaggle` CLI for Kaggle datasets
- [ ] Choose your first dataset source
- [ ] Add dataset with appropriate preprocessing
- [ ] Create a model with conservative parameters
- [ ] Setup training without Mandelbulb initially
- [ ] Run training and monitor progress
- [ ] Experiment with Mandelbulb augmentation
- [ ] Scale up to larger datasets and models

---

**The Gyroidic Dataset Ingestion System embodies the principles of structural honesty, mathematical integrity, and non-teleological learning. It provides a principled approach to dataset ingestion and training that respects the anti-lobotomy constraints while enabling powerful geometric augmentation and evolutionary learning.**

*"We do not optimize toward a target. We create conditions for survivorship and let structure emerge through pressure and selection."*