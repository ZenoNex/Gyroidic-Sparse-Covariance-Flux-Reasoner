import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.fgrt_trainer import FGRTStructuralTrainer
from src.topology.gyroid_differentiation import GyroidFlowConstraint

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4) # 4D for gluing
    
    def forward(self, x):
        return self.fc(x)

def verify():
    print("Verifying Gyroid Integration...")
    
    # 1. Test GyroidFlowConstraint directly
    print("\n[1] Testing GyroidFlowConstraint Primitive...")
    constraint = GyroidFlowConstraint()
    residue = torch.randn(5, 10, requires_grad=True)
    # create embedding that flows perpendicular to Gyroid? Hard to craft manually.
    # Just check it runs.
    embedding = torch.randn(5, 4, requires_grad=True)
    
    stats = constraint(residue, embedding)
    print("Stats keys:", stats.keys())
    print("Dot Product Mean:", stats['dot_product'].mean().item())
    print("Constraint Check Passed (as expected likely False for random):", stats['is_satisfied'].all().item())
    
    # 2. Test Trainer Integration
    print("\n[2] Testing FGRTStructuralTrainer Integration...")
    model = SimpleModel()
    trainer = FGRTStructuralTrainer(model)
    
    input_data = torch.randn(3, 10, requires_grad=True)
    
    print("Running training step...")
    try:
        trainer.train_step(input_data)
        print("Training step completed successfully.")
    except Exception as e:
        print(f"Training step failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
