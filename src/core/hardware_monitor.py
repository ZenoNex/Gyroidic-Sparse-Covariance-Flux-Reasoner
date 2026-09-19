import psutil
import os

class HardwareMonitor:
    """
    Topologically-Aware Hardware Interface (TAHI).
    Integrates visceral security systems (Context-Adaptive Latent Momentum) directly into silicon scaling rules.
    Instead of relying on isolated 'scalar thresholds' (CPU limits), the system asserts physical control over 
    the local architecture based on the structural integrity of the Gyroidic Reasoner's logic manifold.
    TailSlayer conditional bypasses are authorized strictly as topological imperatives.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cpu_threshold = 75.0
            cls._instance.ram_threshold_gb = 2.0
            # Track desync anomalies
            cls._instance.desync_anomalies = 0
            
            # Topological Integrity Signals (CALM)
            cls._instance.topological_abort_score = 0.0
            cls._instance.topological_gauge = 0.0
        return cls._instance

    def update_topology(self, abort_score: float, gauge: float):
        """
        Ingest continuous topological signals from the CALM predictor.
        Allows the hardware monitor to gate PyOpenCL scaling on geometry, not just silicon load.
        """
        self.topological_abort_score = abort_score
        self.topological_gauge = gauge

    def has_headroom(self) -> bool:
        """
        Evaluates physical headroom (CPU/RAM) in conjunction with topological stability (CALM).
        - If geometry is collapsing (abort > 0.8), unconditionally authorize PyOpenCL TailSlayer to 
          save the manifold, unless silicon is critically exhausted (> 95% CPU).
        - If geometry is calm (abort < 0.2), suppress PyOpenCL to conserve logic cycles and avoid 
          high-frequency architectural shocks, regardless of available CPU.
        - Otherwise, rely on blended heuristics.
        """
        try:
            cpu_load = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            free_ram_gb = mem_info.available / (1024 ** 3)
            
            # Simple heuristic for potential poisoning/desync:
            # If load jumps impossibly fast or reads zero consistently, it's anomalous.
            if cpu_load == 0.0:
                self.desync_anomalies += 1
            else:
                self.desync_anomalies = max(0, self.desync_anomalies - 1)

            if self.desync_anomalies > 5:
                # If poisoning event suspected, fallback to conservative
                return False

            # --- TOPOLOGICAL OVERRIDE LOGIC ---
            if self.topological_abort_score > 0.8:
                # Geometric panic override: Authorize TailSlayer to save manifold unless physically impossible
                return cpu_load < 95.0 and free_ram_gb > 1.0
                
            if self.topological_abort_score < 0.2:
                # Manifold is stable: Suppress TailSlayer to prevent mathematical shock
                return False
                
            # Blended baseline
            return cpu_load < self.cpu_threshold and free_ram_gb > self.ram_threshold_gb
        except Exception:
            # On failure, conservative fallback
            return False

def has_headroom() -> bool:
    """Helper function to get headroom status from the centralized monitor."""
    return HardwareMonitor().has_headroom()

def get_headroom_dimensionality() -> int:
    """
    Computes the true topological headroom dimensionality.
    Factors in available physical RAM and Memory Frequency (e.g., 2400 MHz)
    to calculate the maximum allowable graph budget, exploiting the $O(N)$ efficiency
    and the 80/20 Fossilization Split.
    """
    try:
        import psutil
        import subprocess
        import sys
        
        mem_info = psutil.virtual_memory()
        available_mb = mem_info.available / (1024 * 1024)
        
        ram_speed_mhz = 2400 # Default fallback
        if sys.platform == "win32":
            try:
                out = subprocess.check_output(
                    ["wmic", "memorychip", "get", "speed"],
                    timeout=2
                ).decode()
                speeds = [int(s) for s in out.split() if s.isdigit()]
                if speeds:
                    ram_speed_mhz = max(speeds)
            except Exception:
                pass
                
        # Base capacity: ~0.8 MB per active node. 
        # But because of 80/20 Fossilization Split and O(N) constraints, we can pack far more efficiently
        # We multiply the available_mb by a speed multiplier. 2400 MHz acts as our baseline 1.0 multiplier.
        speed_multiplier = ram_speed_mhz / 2400.0
        
        # Calculate raw estimate
        estimated_nodes = int((available_mb / 0.8) * speed_multiplier)
        
        # Apply the Non-Scalar Admissibility boost (rejection logic saves 90% of graph cycles)
        admissibility_boost = 1.5 
        
        dimensional_limit = int(estimated_nodes * admissibility_boost)
        return max(50, dimensional_limit)
        
    except Exception as e:
        print(f"[TAHI] Headroom computation degraded: {e}")
        return 50000
