import time
import os
import glob
import psutil
import torch
import warnings

warnings.filterwarnings('ignore')

# Set low priority
try:
    p = psutil.Process(os.getpid())
    if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(10)
except Exception:
    pass

def run_background_topology_analysis():
    print("Starting background sterile data reruns...")
    # Find semistable data files (e.g., pt files in datasets or data)
    data_files = glob.glob('data/**/*.pt', recursive=True) + glob.glob('datasets/**/*.pt', recursive=True)
    
    if not data_files:
        print("No historical data files found.")
        return
        
    for df in data_files:
        # Wait for PC to not be taxed
        while True:
            cpu = psutil.cpu_percent(interval=1.0)
            if cpu < 30.0:
                print(f"CPU at {cpu}%. Proceeding with {df}")
                break
            time.sleep(5)
            
        try:
            print(f"Running topology analysis on {df}...")
            # Load the data
            data = torch.load(df, map_location='cpu')
            # Mock full topology analysis (in reality we would route it through GyroidReasoner)
            if isinstance(data, torch.Tensor):
                mean_val = data.mean().item()
                print(f"Topological mean scalar: {mean_val}")
            print(f"Successfully processed {df}")
        except Exception as e:
            print(f"Failed to process {df}: {e}")
            
if __name__ == '__main__':
    run_background_topology_analysis()
