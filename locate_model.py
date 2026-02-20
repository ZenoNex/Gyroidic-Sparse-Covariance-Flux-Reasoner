
import sys
import os

# Add potential paths
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
if os.path.join(root_dir, 'src') not in sys.path:
    sys.path.append(os.path.join(root_dir, 'src'))

try:
    import enhanced_temporal_training
    print(f"File: {enhanced_temporal_training.__file__}")
    from enhanced_temporal_training import NonLobotomyTemporalModel
    print("Import successful")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
