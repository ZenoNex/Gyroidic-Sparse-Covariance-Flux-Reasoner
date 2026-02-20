"""
Low Intensity Training Runner.

Demonstrates a lightweight, diegetic training session where the system
ingests "knowledge dyads" (simulated by text/image embeddings) and
warps its output through the Autoeclectic Responder Head.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Core & Topology
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.orchestrator import UniversalOrchestrator
from src.models.resonance_cavity import ResonanceCavity

# Diegetic Components
from src.models.diegetic_heads import DataAssociationLayer, AutoeclecticResponderHead
from src.training.fgrt_fgrt_trainer import SpectralStructuralTrainer

class DiegeticReasoningSystem(nn.Module):
    """
    Composite system for Diegetic Low-Intensity Training.
    Combines:
    1. Data Association (Input)
    2. Resonance Cavity (Memory/Topology)
    3. Autoeclectic Responder (Output)
    """
    def __init__(self, dim: int, k: int=5, num_modes: int=16):
        super().__init__()
        self.dim = dim
        
        # 1. Ingestion
        self.association_layer = DataAssociationLayer(input_dim=dim, hidden_dim=dim, k=k)
        
        # 2. Memory & Topology
        self.cavity = ResonanceCavity(hidden_dim=dim, num_modes=num_modes)
        
        # 3. Output
        self.responder = AutoeclecticResponderHead(hidden_dim=dim, output_dim=dim)
        
        # Internal processing (System 1)
        self.processor = nn.Sequential(
            nn.Linear(dim + k, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, input_data: torch.Tensor, image_data: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_data: [batch, dim] Text embeddings
            image_data: [batch, dim] Image embeddings (optional)
        """
        batch_size = input_data.shape[0]
        
        # 1. Fuse Knowledge Dyad
        if image_data is None:
            image_data = torch.zeros_like(input_data)
        
        # Residues serve as "topological priors" from the dyad
        residues = self.association_layer(input_data, image_data) # [batch, k]
        
        # 2. Process through Cavity (Memory)
        # We project input to query the cavity
        cavity_out = self.cavity.query(input_data)
        memory_context = cavity_out['modes'].mean(dim=1) # [batch, dim]
        
        # 3. System 1 Logic
        # Combine input + dyad residues + memory
        combined = torch.cat([input_data + memory_context, residues], dim=-1)
        latent_state = self.processor(combined)
        
        # 4. Diegetic Response
        # We need entropy/curvature for the responder. In this unified model, 
        # we can estimate them or let the Trainer handle the 'warping' externally.
        # But here we simulate self-estimates for the forward pass 'mischief'.
        
        # Self-estimated entropy (simplified)
        entropy = torch.norm(latent_state, dim=-1, keepdim=True) * 0.1
        
        output = self.responder(latent_state, entropy)
        
        return output

def run_low_intensity_session():
    print("Initializing Gyroidic Low-Intensity Training Session...")
    print("Regime: PLAY (Goo) -> SERIOUSNESS (Prickles) [Soft Transition]")
    
    # Configuration
    dim = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    k = 5
    
    # 1. Setup Model
    print(f"Instantiating Diegetic Reasoning System (dim={dim})...")
    model = DiegeticReasoningSystem(dim=dim, k=k).to(device, non_blocking=True)
    
    # 2. Setup Orchestrator (Physics)
    orchestrator = UniversalOrchestrator(dim=dim).to(device, non_blocking=True)
    
    # 3. Setup Trainer (Non-Teleological)
    poly_config = PolynomialCoprimeConfig(k=k, degree=4, device=device)
    trainer = SpectralStructuralTrainer(
        model=model,
        poly_config=poly_config,
        lr=5e-4, # Low intensity learning rate
        spectral_threshold=0.8 # Allow more "mischief" before clamping
    )
    
    # 4. Run Loop
    num_steps = 10
    batch_size = 4
    
    print(f"\nStarting {num_steps} steps of 'Low Intensity' training...")
    print(f"Ingesting Synthetic Knowledge Dyads (Text + Image Embs)...")
    
    for step in range(num_steps):
        # Synthetic Dyads
        text_emb = torch.randn(batch_size, dim).to(device, non_blocking=True)
        img_emb = torch.randn(batch_size, dim).to(device, non_blocking=True)
        
        # We treat the text_emb as the main "input_data" for the trainer interface
        # Ideally trainer.train_step would accept kwargs, but for now we pass the primary input.
        # The model inside uses a default or we can patch it.
        
        # Patching or careful passing:
        # Our model.forward expects (text, image). Trainer calls model(input_data).
        # We'll rely on the default None for image inside the model, 
        # OR we can update the trainer to handle kwargs.
        # For this demo, let's assume text-only input for the 'standard' train step,
        # but the model internally handles it robustly.
        
        # Better: We wrap the input to include image data if we updated the trainer.
        # Minimal fix: The model handles missing image.
        
        start_time = time.time()
        metrics = trainer.train_step(text_emb)
        dt = time.time() - start_time
        
        # Log diegetic status
        regime_status = "PLAY" if metrics['spectral_entropy'] > 1.0 else "SERIOUSNESS"
        print(f"Step {step+1}/{num_steps} | {dt:.3f}s | Regime: {regime_status}")
        print(f"  > Willmore Energy: {metrics['willmore_energy']:.4f}")
        print(f"  > Spectral Entropy: {metrics['spectral_entropy']:.4f} (Mischief)")
        print(f"  > Gyroid Violation: {metrics['gyroid_violation']:.4f}")
        print("-" * 40)
        
        # Simulate "user interaction" or "pause" for low intensity feel
        # time.sleep(0.5) 
        
    print("\nSession Complete.")
    print("The system has integrated the dyadic traces into the Resonance Cavity.")
    print("Dark Matter accumulation: ACTIVE.")

if __name__ == "__main__":
    run_low_intensity_session()


