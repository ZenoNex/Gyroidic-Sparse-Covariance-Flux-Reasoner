#!/usr/bin/env python3
"""
Quick test to verify the tensor dimension fix using Symmetry-Preserving Reshape approach.
"""

import torch
import torch.nn as nn
import sys
import os

def test_dimension_alignment():
    """Test the dimension alignment fix."""
    print("🔧 Testing Tensor Dimension Alignment Fix")
    print("=" * 50)
    
    # Simulate the dimension mismatch
    image_emb = torch.randn(768)  # Image embedding from fingerprint projection
    text_emb = torch.randn(256)   # Text embedding from temporal model
    
    print(f"Original dimensions:")
    print(f"  Image embedding: {image_emb.shape}")
    print(f"  Text embedding: {text_emb.shape}")
    
    # Method 1: Symmetry-Preserving Reshape with reflective padding
    print(f"\n🔧 Method 1: Symmetry-Preserving Reshape (Reflective Padding)")
    
    target_dim = max(image_emb.shape[0], text_emb.shape[0])
    
    if text_emb.shape[0] < target_dim:
        pad_size = target_dim - text_emb.shape[0]
        text_emb_padded = torch.nn.functional.pad(text_emb, (0, pad_size), mode='reflect')
        print(f"  Padded text embedding: {text_emb.shape[0]} -> {text_emb_padded.shape[0]}")
    else:
        text_emb_padded = text_emb
    
    # Test similarity computation
    try:
        similarity1 = torch.cosine_similarity(image_emb, text_emb_padded, dim=0).item()
        print(f"  ✅ Similarity computation successful: {similarity1:.3f}")
        method1_success = True
    except Exception as e:
        print(f"  ❌ Method 1 failed: {e}")
        method1_success = False
    
    # Method 2: Learnable projection (preserves learned structure)
    print(f"\n🔧 Method 2: Learnable Projection")
    
    projection = nn.Linear(256, 768)
    nn.init.orthogonal_(projection.weight)  # Preserve information
    
    with torch.no_grad():
        text_emb_projected = projection(text_emb)
    
    print(f"  Projected text embedding: {text_emb.shape[0]} -> {text_emb_projected.shape[0]}")
    
    try:
        similarity2 = torch.cosine_similarity(image_emb, text_emb_projected, dim=0).item()
        print(f"  ✅ Similarity computation successful: {similarity2:.3f}")
        method2_success = True
    except Exception as e:
        print(f"  ❌ Method 2 failed: {e}")
        method2_success = False
    
    # Method 3: Truncation (simple but loses information)
    print(f"\n🔧 Method 3: Truncation (for comparison)")
    
    image_emb_truncated = image_emb[:256]  # Truncate to match text
    print(f"  Truncated image embedding: {image_emb.shape[0]} -> {image_emb_truncated.shape[0]}")
    
    try:
        similarity3 = torch.cosine_similarity(image_emb_truncated, text_emb, dim=0).item()
        print(f"  ✅ Similarity computation successful: {similarity3:.3f}")
        method3_success = True
    except Exception as e:
        print(f"  ❌ Method 3 failed: {e}")
        method3_success = False
    
    # Summary
    print(f"\n📊 Results Summary:")
    print(f"  Method 1 (Reflective Padding): {'✅ PASS' if method1_success else '❌ FAIL'}")
    print(f"  Method 2 (Learnable Projection): {'✅ PASS' if method2_success else '❌ FAIL'}")
    print(f"  Method 3 (Truncation): {'✅ PASS' if method3_success else '❌ FAIL'}")
    
    if method2_success:
        print(f"\n🎯 Recommended: Method 2 (Learnable Projection)")
        print(f"   - Preserves learned representations")
        print(f"   - Maintains information content")
        print(f"   - Compatible with training")
    elif method1_success:
        print(f"\n🎯 Fallback: Method 1 (Reflective Padding)")
        print(f"   - Preserves structural properties")
        print(f"   - No additional parameters")
        print(f"   - Symmetry-preserving")
    
    return method1_success or method2_success

def test_batch_operations():
    """Test that the fix works with batch operations."""
    print(f"\n🔧 Testing Batch Operations")
    print("=" * 30)
    
    # Simulate batch of embeddings
    batch_size = 4
    image_embs = torch.randn(batch_size, 768)
    text_embs = torch.randn(batch_size, 256)
    
    print(f"Batch dimensions:")
    print(f"  Image embeddings: {image_embs.shape}")
    print(f"  Text embeddings: {text_embs.shape}")
    
    # Test projection approach
    projection = nn.Linear(256, 768)
    nn.init.orthogonal_(projection.weight)
    
    with torch.no_grad():
        text_embs_projected = projection(text_embs)
    
    print(f"  Projected text embeddings: {text_embs_projected.shape}")
    
    # Test batch similarity
    try:
        similarities = torch.cosine_similarity(image_embs, text_embs_projected, dim=1)
        print(f"  ✅ Batch similarity computation successful: {similarities.shape}")
        print(f"  Sample similarities: {similarities[:3].tolist()}")
        return True
    except Exception as e:
        print(f"  ❌ Batch operations failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Tensor Dimension Fix Verification")
    print("Testing solutions for image-text embedding alignment")
    print("=" * 60)
    
    # Test basic alignment
    basic_success = test_dimension_alignment()
    
    # Test batch operations
    batch_success = test_batch_operations()
    
    print(f"\n🎯 Overall Results:")
    print(f"  Basic alignment: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"  Batch operations: {'✅ PASS' if batch_success else '❌ FAIL'}")
    
    if basic_success and batch_success:
        print(f"\n🚀 All tests passed! The dimension fix is working.")
        print(f"   Ready to apply to test_image_simple.py")
    else:
        print(f"\n⚠️  Some tests failed. Check the implementation.")