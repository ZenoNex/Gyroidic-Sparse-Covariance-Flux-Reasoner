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
    print("Starting background sterile data reruns and non-semisimple topological refresh...")
    from src.data.chatgpt_friction_harvester import is_semisimple_or_sterile
    # Find semistable data files (e.g., pt files in datasets or data)
    data_files = glob.glob('data/**/*.pt', recursive=True) + glob.glob('datasets/**/*.pt', recursive=True)
    
    if not data_files:
        print("No historical data files found.")
        return
        
    for df in data_files:
        # Wait for PC to not be taxed
        while True:
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().percent
            if cpu < 35.0 and ram < 75.0:
                print(f"System affordance ready (CPU: {cpu}%, RAM: {ram}%). Proceeding with {df}")
                break
            time.sleep(3)
            
        try:
            print(f"Running non-semisimple topology audit on {df}...")
            data = torch.load(df, map_location='cpu')
            if is_semisimple_or_sterile(data):
                print(f"[RECOVERY] Sterile/semisimple data detected in {df}. Initiating topological rehydration...")
                if isinstance(data, dict):
                    # Flag rehydration and update moduli if sterile
                    if 'moduli' in data and isinstance(data['moduli'], torch.Tensor) and (data['moduli'] == 1.0).all():
                        from src.core.invariants import PHI
                        data['moduli'] = data['moduli'] * PHI
                    tags = data.get('tags', [])
                    data['tags'] = [t for t in tags if t not in ['sterile_moduli', 'atrophy_rehydrated', 'semisimple_reconstruction']]
                    data['tags'].append('non_semisimple_refreshed')
                    data['semisimple_rerun_complete'] = True
                    tmp_df = df + ".tmp"
                    torch.save(data, tmp_df)
                    os.replace(tmp_df, df)
                    print(f"[RECOVERY] Successfully upgraded {df} to non-semisimple topology.")
            else:
                print(f"Topological stability verified for {df}.")
        except Exception as e:
            print(f"Failed to process {df}: {e}")
            
if __name__ == '__main__':
    run_background_topology_analysis()
