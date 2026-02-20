# Tensor Dimension Fix Guide

**How to handle dimension mismatches while preserving learning**

This guide explains how to fix tensor dimension mismatches in the Gyroidic AI System without breaking the learning process or violating anti-lobotomy principles.

---

## 🚨 The Problem

When integrating different parts of the AI system, you might encounter errors like:

```
RuntimeError: The size of tensor a (768) must match the size of tensor b (256) at non-singleton dimension 0
```

This happens when:
- **Image embeddings** are 768-dimensional (from fingerprint projection)
- **Text embeddings** are 256-dimensional (from temporal model hidden state)
- **Operations** like `torch.cosine_similarity()` require matching dimensions

---

## 🔧 The Solution: Learnable Projection

### ✅ Recommended Approach
Use a **learnable projection** that preserves the learned representations:

```python
# Create projection layer
text_to_image_projection = torch.nn.Linear(256, 768)
torch.nn.init.orthogonal_(text_to_image_projection.weight)  # Preserve information

# Project text embedding to image space
with torch.no_grad():
    text_emb_projected = text_to_image_projection(text_emb)  # [256] -> [768]

# Now compute similarity
similarity = torch.cosine_similarity(image_emb, text_emb_projected, dim=0)
```

### Why This Works
1. **Preserves learned structure** from the temporal model
2. **Maintains information content** through orthogonal initialization
3. **Compatible with training** - can be learned if needed
4. **Follows anti-lobotomy principles** - doesn't oversimplify

---

## 🔄 Symmetry-Preserving Reshape (Recommended)

### When to Use
This is the **established solution** used throughout the Gyroidic system for tensor dimension mismatches.

```python
# Apply Symmetry-Preserving Reshape using reflective padding
if tensor.shape[0] != target_size:
    if tensor.shape[0] < target_size:
        pad_size = target_size - tensor.shape[0]
        tensor_padded = torch.nn.functional.pad(tensor, (0, pad_size), mode='reflect')
        print(f"🔧 Applied Symmetry-Preserving padding: {tensor.shape[0]} -> {tensor_padded.shape[0]}")
    else:
        tensor_padded = tensor[:target_size]  # Truncate if larger

# Now use tensor_padded for operations
similarity = torch.cosine_similarity(image_emb, tensor_padded, dim=0)
```

### Why This Works
1. **Preserves structural properties** through reflective padding
2. **No additional parameters** needed
3. **Symmetry-preserving** - maintains topological properties
4. **Proven solution** - used in the core breakthrough architecture
5. **Follows anti-lobotomy principles** - documented in BREAKTHROUGH_REPORT.md

### Mathematical Foundation
Based on the breakthrough solution: **64 → 65 dimensions using `torch.nn.functional.pad(mode='reflect')`**
- Preserves Chiral Torsion while enabling valid reshape operations
- Used throughout the system for Bezout, Chern-Simons, Soliton, and Soft Gates compatibility

---

## ❌ What NOT to Do

### Don't Truncate
```python
# BAD: Loses information
image_emb_truncated = image_emb[:256]
```

### Don't Zero-Pad
```python
# BAD: Adds meaningless zeros
text_emb_padded = torch.nn.functional.pad(text_emb, (0, 512), mode='constant', value=0)
```

### Don't Change Model Architecture
```python
# BAD: Breaks existing learning
text_model = NonLobotomyTemporalModel(hidden_dim=768)  # Don't change this
```

---

## 🧪 Testing Your Fix

### Quick Test
```python
# Test dimension alignment
image_emb = torch.randn(768)
text_emb = torch.randn(256)

# Apply your fix here
# ...

# This should work without errors
similarity = torch.cosine_similarity(image_emb, text_emb_fixed, dim=0)
print(f"Similarity: {similarity:.3f}")
```

### Batch Test
```python
# Test with batches
image_embs = torch.randn(4, 768)
text_embs = torch.randn(4, 256)

# Apply your fix here
# ...

# This should work for batches
similarities = torch.cosine_similarity(image_embs, text_embs_fixed, dim=1)
print(f"Batch similarities: {similarities}")
```

---

## 🎯 Implementation Examples

### In Training Code
```python
class MultimodalTrainer:
    def __init__(self):
        # Create projection for dimension alignment
        self.text_to_image_proj = nn.Linear(256, 768)
        nn.init.orthogonal_(self.text_to_image_proj.weight)
    
    def compute_cross_modal_loss(self, image_embs, text_embs):
        # Align dimensions
        text_embs_aligned = self.text_to_image_proj(text_embs)
        
        # Compute loss
        similarities = torch.cosine_similarity(image_embs, text_embs_aligned, dim=1)
        return -similarities.mean()  # Maximize similarity
```

### In Inference Code
```python
def compare_image_text(image_emb, text_emb):
    # Create projection (could be loaded from training)
    projection = nn.Linear(text_emb.shape[0], image_emb.shape[0])
    nn.init.orthogonal_(projection.weight)
    
    # Align and compare
    with torch.no_grad():
        text_emb_aligned = projection(text_emb)
        similarity = torch.cosine_similarity(image_emb, text_emb_aligned, dim=0)
    
    return similarity.item()
```

---

## 📚 Related Documentation

### Core Concepts
- [Breakthrough Report](docs/BREAKTHROUGH_REPORT.md) - Original Symmetry-Preserving Reshape
- [Philosophy](docs/PHILOSOPHY.md) - Anti-lobotomy principles
- [Implementation Guide](docs/IMPLEMENTATION_INTEGRITY_GUIDE.md) - Code standards

### Practical Guides
- [User Manual](USER_MANUAL.md) - How to use the system
- [Getting Started](GETTING_STARTED.md) - First steps
- [Command Reference](COMMAND_REFERENCE.md) - Quick lookup

---

## 🔬 Technical Details

### Why Dimensions Mismatch
1. **Image fingerprints** are 137-dimensional (color + texture + edges)
2. **Image embeddings** are projected to 768-dimensional space for compatibility
3. **Text embeddings** start as 768-dimensional but get processed by temporal model
4. **Temporal model** has 256-dimensional hidden state for efficiency
5. **Cross-modal operations** require matching dimensions

### Mathematical Foundation
The learnable projection approach is based on:
- **Information preservation** through orthogonal initialization
- **Learned alignment** that can adapt during training
- **Topological consistency** with the embedding spaces
- **Anti-lobotomy compliance** by maintaining complexity

### Performance Considerations
- **Learnable projection**: Adds parameters but preserves information
- **Reflective padding**: No parameters but may introduce artifacts
- **Computational cost**: Minimal for both approaches
- **Memory usage**: Slightly higher for projection approach

---

This fix ensures that the AI system can handle multimodal data without breaking the learning process or violating the architectural principles that make it effective.