"""
Verification script for Diegetic Responder Heads and Data Association.
"""

import torch
from src.models.diegetic_heads import DataAssociationLayer, AutoeclecticResponderHead

def verify_diegetic_system():
    print("Verifying Diegetic System...")
    
    batch_size = 2
    dim = 64
    k = 5
    out_dim = 128
    
    # 1. Test Data Association (Knowledge Dyads)
    print("Testing Data Association (Knowledge Dyads)...")
    text_emb = torch.randn(batch_size, dim)
    image_emb = torch.randn(batch_size, dim)
    
    association_layer = DataAssociationLayer(input_dim=dim, hidden_dim=64, k=k)
    residues = association_layer(text_emb, image_emb)
    
    print(f"Residue shape: {residues.shape}")
    assert residues.shape == (batch_size, k)
    
    # 2. Test Autoeclectic Responder Head
    print("Testing Autoeclectic Responder Head...")
    state = torch.randn(batch_size, dim)
    entropy = torch.rand(batch_size, 1)
    curvature = torch.rand(batch_size, 1)
    
    responder = AutoeclecticResponderHead(hidden_dim=dim, output_dim=out_dim)
    output = responder(state, entropy, curvature)
    
    print(f"Responder output shape: {output.shape}")
    assert output.shape == (batch_size, out_dim)
    
    # 3. Test High-Entropy Warping (Mischief)
    print("Testing High-Entropy Warping...")
    low_entropy = torch.zeros(batch_size, 1)
    high_entropy = torch.ones(batch_size, 1) * 2.0
    
    out_low = responder(state, low_entropy)
    out_high = responder(state, high_entropy)
    
    diff = torch.norm(out_low - out_high)
    print(f"Entropy-driven difference norm: {diff.item()}")
    assert diff > 0.0, "Output should vary with entropy levels (mischief effect)"
    
    print("Diegetic System Verification PASSED.")

if __name__ == "__main__":
    verify_diegetic_system()
