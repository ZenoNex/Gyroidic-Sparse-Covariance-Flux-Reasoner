# Getting Started with Gyroidic AI

**Your first 15 minutes with the Gyroidic AI System**

This guide will get you up and running with the AI system in just a few minutes.

---

## 🚀 Step 1: Install (2 minutes)

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Test Everything Works
```bash
python test_fixes_verification.py
```

You should see:
```
✅ PIL Compatibility: PASS
✅ pas_h Type Handling: PASS  
✅ Simple Integration: PASS
🚀 All fixes verified! Ready to proceed with image integration.
```

---

## 💬 Step 2: Start Chatting (3 minutes)

### Launch the Web Interface
```bash
python src/ui/diegetic_terminal.py
```

### Open Your Browser
Go to: **http://localhost:8000**

### Try These Messages
- "hello" - Basic greeting
- "what can you do?" - Learn about capabilities
- "tell me about physics" - Test knowledge
- Drag an image into the chat - Test image analysis

---

## 📚 Step 3: Train on Your First Dataset (10 minutes)

### Quick Training on Movie Reviews
```bash
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3
```

This will:
1. Download movie reviews
2. Process them into the AI's format
3. Train the AI to understand sentiment and opinions
4. Take about 5-10 minutes

### Check What Happened
```bash
python dataset_command_interface.py status
```

### Test the Results
Go back to the chat interface and ask:
- "What do you think about movies?"
- "Are you positive or negative about this: 'The movie was amazing!'"

---

## 🌟 What You Just Accomplished

### ✅ You Have a Working AI System
- The AI can chat with you coherently
- It can analyze images you upload
- It learned from 500 movie reviews
- It understands sentiment and opinions

### ✅ You Can Train It on Anything
- Wikipedia articles
- Your own documents
- Scientific papers
- Programming code
- Any text data

---

## 🎯 What to Try Next

### Learn from Wikipedia (5 minutes)
```bash
# Teach it physics
python dataset_command_interface.py add-wikipedia --topics physics --samples 300 --train

# Teach it mathematics  
python dataset_command_interface.py add-wikipedia --topics mathematics --samples 200 --train
```

### Train on Your Documents
```bash
# Put your files in a folder called 'documents'
python dataset_command_interface.py train-local --path ./documents/ --epochs 5
```

### Try Different Datasets
```bash
# Question-answering
python dataset_command_interface.py quick-start --dataset squad --samples 500 --epochs 3

# Scientific papers
python dataset_command_interface.py quick-start --dataset arxiv --samples 300 --epochs 5

# Programming code
python dataset_command_interface.py quick-start --dataset codeparrot --samples 400 --epochs 4
```

---

## 📖 Available Datasets

### Popular Options
- **imdb** - Movie reviews (opinions, sentiment)
- **squad** - Question-answering (facts, knowledge)
- **wikitext** - Wikipedia articles (general knowledge)
- **arxiv** - Scientific papers (technical knowledge)
- **codeparrot** - Programming code (coding skills)

### Wikipedia Topics
- **physics** - Quantum mechanics, relativity, thermodynamics
- **mathematics** - Linear algebra, calculus, topology
- **computer_science** - Machine learning, algorithms
- **philosophy** - Logic, ethics, philosophy of mind
- **biology** - Evolution, genetics, molecular biology

---

## ⚙️ Understanding the Options

### `--samples` (How Much Data)
- **500** - Quick training, good for testing
- **1000** - Good balance of speed and quality
- **2000+** - Better results but takes longer

### `--epochs` (How Long to Train)
- **3** - Quick learning, basic understanding
- **5-8** - Good learning, recommended
- **10+** - Deep learning, best results

### `--augment` (Data Expansion)
- Automatically creates more training examples
- Good for small datasets
- Uses advanced mathematical techniques

---

## 💾 Storage Management

### How Much Space Things Use
- **500 samples**: ~45-80 MB
- **1000 samples**: ~90-160 MB
- **2000 samples**: ~180-320 MB

### Check Your Usage
```bash
python dataset_command_interface.py status
```

### The System is Designed for 100GB
You can train on many datasets before running out of space.

---

## 🚨 If Something Goes Wrong

### "PIL.Image has no attribute 'Resampling'"
```bash
pip install --upgrade Pillow
```

### "Backend Connectivity Lost"
```bash
# Make sure the backend is running
python src/ui/diegetic_backend.py
```

### "Out of storage space"
```bash
# Check usage
python dataset_command_interface.py status

# Use smaller datasets
python dataset_command_interface.py quick-start --dataset imdb --samples 300 --epochs 3
```

### "Training too slow"
```bash
# Use fewer samples and epochs
python dataset_command_interface.py quick-start --dataset imdb --samples 500 --epochs 3
```

---

## 🎓 Ready for More?

### Complete Guides
- [User Manual](USER_MANUAL.md) - Everything you can do
- [Command Reference](COMMAND_REFERENCE.md) - Quick command lookup
- [Dataset Guide](DATASET_INGESTION_GUIDE.md) - Complete training guide

### Advanced Features
- Train on multiple datasets simultaneously
- Use advanced data augmentation
- Create custom model architectures
- Process images and text together

### Research Features
- Anti-lobotomy architecture (prevents oversimplification)
- Topological reasoning (mathematical coherence)
- Evolutionary learning (non-gradient methods)
- Efficient multimodal processing

---

## 🌟 What Makes This Special

This isn't just another AI system. It has breakthrough features:

### ✅ Coherent Responses
No garbled output - generates human-like text

### ✅ True Learning
Actually learns from new data, doesn't just retrieve

### ✅ Storage Efficient
Works within 100GB, practical for personal use

### ✅ Anti-Lobotomy
Maintains complexity, doesn't oversimplify

### ✅ Multimodal
Handles both text and images together

---

## 🚀 You're Ready!

You now have:
- A working AI chat system
- The ability to train on any dataset
- Understanding of the basic commands
- Knowledge of what to try next

Start experimenting and see what this AI can learn! The system is designed to be both powerful and easy to use.

**Happy training!** 🎉