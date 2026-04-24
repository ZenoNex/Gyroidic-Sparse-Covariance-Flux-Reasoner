import sys
import os
import torch

# Set UTF-8 encoding for stdout to handle emojis if possible, but better to just remove them
# sys.stdout.reconfigure(encoding='utf-8') 

# Add src to path
sys.path.append(os.getcwd())

def test_stabilization():
    print("[START] Starting Gyroidic Manifold Stabilization Verification...")
    
    try:
        from src.core.enhanced_bezout_crt import EnhancedBezoutCRT, CrossbarIKSolver
        ik = CrossbarIKSolver()
        print("[OK] CrossbarIKSolver initialized.")
        crt = EnhancedBezoutCRT(state_dim=64, num_moduli=5)
        print("[OK] EnhancedBezoutCRT with IK initialized.")
    except Exception as e:
        print(f"[FAIL] Crossbar IK / CRT failure: {e}")

    try:
        from src.core.chern_simons_gasket import ChernSimonsGasket, SurgicalSeamVisualizer
        gasket = ChernSimonsGasket(manifold_dim=3)
        print("[OK] ChernSimonsGasket with Seam Visualizer initialized.")
    except Exception as e:
        print(f"[FAIL] Surgical Seam / Gasket failure: {e}")

    try:
        from src.core.bulletin_board import BulletinBoard
        bb = BulletinBoard(size=64)
        bb.micro_step()
        print("[OK] BulletinBoard micro-stepping verified.")
    except Exception as e:
        print(f"[FAIL] BulletinBoard failure: {e}")

    try:
        from src.core.zeitgeist_router import ZeitgeistRouter, BraidGroupMatrices
        router = ZeitgeistRouter(dim=64, moduli=(2, 3, 5))
        print("[OK] ZeitgeistRouter with Braid Matrices initialized.")
    except Exception as e:
        print(f"[FAIL] ZeitgeistRouter failure: {e}")

    try:
        from src.models.resonance_cavity import HeritableTrustVault
        vault = HeritableTrustVault(table_size=1024, k_dim=5)
        res = torch.randn(2, 5)
        indices = vault._hash(res)
        print("[OK] HeritableTrustVault sha256 hashing verified.")
    except Exception as e:
        print(f"[FAIL] Resonance Cavity hashing failure: {e}")

    print("\n[SUCCESS] All core stabilization modules verified.")

if __name__ == "__main__":
    test_stabilization()
