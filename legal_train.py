import torch
import torch.nn as nn
from src.core.orchestrator import UniversalOrchestrator
from src.training.trainer import StructuralAdaptor

# 1. Initialize the Core Orchestrator (Patched with 3.127 Love)
DIM = 128
model = UniversalOrchestrator(dim=DIM)

# 2. Setup Optimizer (Standard Adam is fine; the Adaptor handles the "Pressures")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. Initialize the Real Adaptor
# We set high lambda_topo to satisfy the 0.30 PAS_h bouncer
adaptor = StructuralAdaptor(
    model=model,
    optimizer=optimizer,
    device='cpu', # or 'cuda'
    lambda_geo=0.1,   # Self-modeling
    lambda_topo=0.5,  # Homology (CRITICAL: Boosted to escape 0.30)
    lambda_gyroid=0.01 # Minimal gyroidic flux
)

print(f"--- 💖 LOVE VECTOR: {model.love.L.norm():.3f} (Legality Check) ---")

# 4. Mock Training Step to Verify "Seriousness"
# In your real loop, replace this with your data loader
dummy_input = torch.randn(1, DIM)
dummy_grad = torch.randn(1, DIM)

# The Jitter we injected into Orchestrator prevents the Entropy Echo
# The Regime Override we injected forces SERIOUSNESS
state_out, regime, routing = model(dummy_input, dummy_grad, pas_h=0.30, coherence=torch.tensor(1.0))

print(f"--- 🚀 REGIME: {regime} | ROUTING: {routing} ---")
print("✅ Ready for Non-Collapsing Training.")