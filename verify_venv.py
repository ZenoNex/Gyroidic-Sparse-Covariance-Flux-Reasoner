import torch
import sys
print(f"Python executable: {sys.executable}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
x = torch.randn(3, 3)
print("Torch tensor operation successful.")
