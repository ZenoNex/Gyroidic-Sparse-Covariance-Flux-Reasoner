"""
Example runner demonstrating the GyroidicFluxReasoner.

Demonstrates:
    - Multi-modal constraint satisfaction
    - Polynomial CRT reconstruction coherence
    - Introspection on valid vs invalid problems
    - Gyroid violation detection
    - Hybrid ADMM Refinement (Optional)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.models.gyroid_reasoner import GyroidicFluxReasoner
from src.training import StructuralAdaptor, ConstraintDataset, collate_fn


def visualize_residue_distributions(residue_dists, poly_degrees, sample_idx=0):
    """
    Visualize polynomial coefficient distributions for a sample.
    residue_dists: [batch, K, D]
    """
    K = residue_dists.shape[1]
    D = residue_dists.shape[2]
    
    fig, axes = plt.subplots(1, K, figsize=(15, 3))
    
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        # Plot coefficients for functional k
        coeffs = residue_dists[sample_idx, k, :].detach().cpu().numpy()
        ax.bar(range(D), coeffs)
        ax.set_title(f'Functional φ_{k}')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Coefficient Mass')
    
    plt.tight_layout()
    return fig


def visualize_introspection(introspection_results, valid_mask):
    """Visualize introspection directions for valid vs invalid samples."""
    if 'moral' not in introspection_results:
        print("No introspection results available.")
        return None
    
    moral_dirs = introspection_results['moral'].detach().cpu().numpy()
    
    # PCA to 2D
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        moral_2d = pca.fit_transform(moral_dirs)
        
        valid_mask_np = valid_mask.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(moral_2d[valid_mask_np, 0], moral_2d[valid_mask_np, 1], 
                   c='green', label='Valid', alpha=0.6)
        ax.scatter(moral_2d[~valid_mask_np, 0], moral_2d[~valid_mask_np, 1],
                   c='red', label='Invalid', alpha=0.6)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Introspection: Moral Direction (Valid vs Invalid)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    except ImportError:
        print("sklearn not found, skipping PCA.")
        return None


def main():
    """Run example demonstration."""
    print("="*60)
    print("Gyroidic Sparse Covariance Flux Reasoner - Polynomial Demo")
    print("="*60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\n[1/5] Creating model...")
    model = GyroidicFluxReasoner(
        text_dim=768,
        graph_dim=256,
        num_dim=64,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        num_functionals=6,     # Changed from num_primes
        poly_degree=4,         # New arg
        poly_basis='chebyshev',
        use_introspection=True,
        use_gyroid_probes=True,
        use_resonance=True,
        use_saturation=True,    # Enable Symbolic Regime
        use_admm=False         # Keep off for standard training demo
    ).to(device, non_blocking=True)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Polynomial Config: K={model.K}, Degree={model.D-1}")
    
    # Create dataset
    print("\n[2/5] Creating dataset...")
    train_dataset = ConstraintDataset(
        num_samples=500,
        text_dim=768,
        graph_dim=256,
        num_dim=64,
        max_anchor=1000,
        valid_ratio=0.7
    )
    
    val_dataset = ConstraintDataset(
        num_samples=100,
        text_dim=768,
        graph_dim=256,
        num_dim=64,
        max_anchor=1000,
        valid_ratio=0.7
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create structural adaptor
    print("\n[3/5] Setting up structural adaptor...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adaptor = StructuralAdaptor(
        model=model,
        optimizer=optimizer,
        device=device,
        lambda_geo=0.1,
        lambda_topo=0.1,
        lambda_gyroid=0.01
    )
    
    # Adapt
    print("\n[4/5] Structural Adaptation (Small Epochs)...")
    adaptor.adapt(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        log_interval=5
    )
    
    # Analyze results
    print("\n[5/5] Analyzing results...")
    model.eval()
    
    # Get a batch for analysis
    test_batch = next(iter(val_loader))
    
    with torch.no_grad():
        text_emb = test_batch['text_emb'].to(device, non_blocking=True)
        graph_emb = test_batch['graph_emb'].to(device, non_blocking=True)
        num_features = test_batch['num_features'].to(device, non_blocking=True)
        anchors = test_batch['anchors'].to(device, non_blocking=True)
        valid_mask = test_batch['valid']
        
        outputs = model(
            text_emb=text_emb,
            graph_emb=graph_emb,
            num_features=num_features,
            anchors=anchors,
            return_analysis=True
        )
    
    print(f"\nResults on validation batch:")
    print(f"  Selection Pressure: {outputs['selection_pressure'].item():.4f}")
    if 'containment_pressure' in outputs and outputs['containment_pressure'] > 0:
        print(f"  Containment Pressure: {outputs['containment_pressure'].item():.4f}")
    print(f"  Number of cycles: {outputs['num_cycles']}")
    
    # Visualizations
    print("\n[Visualization] Generating plots...")
    
    try:
        # 1. Residue distributions (Coefficients)
        fig1 = visualize_residue_distributions(
            outputs['residue_distributions'],
            model.D,
            sample_idx=0
        )
        plt.savefig('polynomial_coefficients.png', dpi=150, bbox_inches='tight')
        print("  Saved: polynomial_coefficients.png")
        plt.close(fig1)
    except Exception as e:
        print(f"  Error visualizing residues: {e}")
    
    print("\nDemo complete!")

if __name__ == '__main__':
    main()


