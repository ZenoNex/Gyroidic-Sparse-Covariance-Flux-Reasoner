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
            
            # Thermodynamic Signals (Carnot-Möbius / Energy Monitor)
            cls._instance.thermal_runaway = False
            cls._instance.p_waste = 0.0
            cls._instance.p_cool_max = 1000.0
            cls._instance.eta_stack = 1.0
            cls._instance.thermal_margin = 1.0  # (p_cool_max - p_waste) / p_cool_max
            cls._instance.thermal_source = "none"
        return cls._instance

    def update_topology(self, abort_score: float, gauge: float):
        """
        Ingest continuous topological signals from the CALM predictor.
        Allows the hardware monitor to gate PyOpenCL scaling on geometry, not just silicon load.
        """
        self.topological_abort_score = abort_score
        self.topological_gauge = gauge

    def update_thermodynamics(
        self,
        p_waste: float,
        p_cool_max: float,
        thermal_runaway: bool,
        eta_stack: float = 1.0,
        source: str = "carnot_mobius"
    ):
        """
        Ingest continuous thermodynamic telemetry from CarnotMobiusLedger,
        DiegeticPhysicsEngine, or Cleanroom systems.
        Allows the hardware monitor and compute affordance chain to throttle or gate
        before reaching melting / critical depth N_crit.
        """
        self.p_waste = float(p_waste)
        self.p_cool_max = max(1e-4, float(p_cool_max))
        self.thermal_runaway = bool(thermal_runaway) or (self.p_waste > self.p_cool_max)
        self.eta_stack = float(eta_stack)
        self.thermal_margin = max(0.0, (self.p_cool_max - self.p_waste) / self.p_cool_max)
        self.thermal_source = str(source)

    def has_headroom(self) -> bool:
        """
        Evaluates physical headroom (CPU/RAM) in conjunction with topological stability (CALM)
        and thermodynamic ledger state (Carnot-Möbius).
        - If in thermal runaway, suppress computation to prevent melting/rupture.
        - If geometry is collapsing (abort > 0.8), authorize PyOpenCL TailSlayer to save manifold
          unless silicon is exhausted (> 95% CPU).
        - If geometry is calm (abort < 0.2), suppress PyOpenCL to conserve logic cycles.
        """
        # Thermal runaway override: immediately suppress headroom to cool down
        if self.thermal_runaway:
            return False

        try:
            cpu_load = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            free_ram_gb = mem_info.available / (1024 ** 3)
            
            # Simple heuristic for potential poisoning/desync:
            if cpu_load == 0.0:
                self.desync_anomalies += 1
            else:
                self.desync_anomalies = max(0, self.desync_anomalies - 1)

            if self.desync_anomalies > 5:
                return False

            # --- TOPOLOGICAL OVERRIDE LOGIC ---
            if self.topological_abort_score > 0.8:
                return cpu_load < 95.0 and free_ram_gb > 1.0
                
            if self.topological_abort_score < 0.2:
                return False
                
            # Blended baseline with thermal margin scaling
            effective_cpu_threshold = self.cpu_threshold * max(0.5, self.thermal_margin)
            return cpu_load < effective_cpu_threshold and free_ram_gb > self.ram_threshold_gb
        except Exception:
            return False

def has_headroom() -> bool:
    """Helper function to get headroom status from the centralized monitor."""
    return HardwareMonitor().has_headroom()

def has_compute_affordance() -> bool:
    """
    Evaluates raw physical computational affordance / load headroom.
    Returns True if system CPU and memory are not overloaded AND
    the system is NOT in thermodynamic runaway.
    Used to gate mathematical refinements (SOC, fractal zones, fauna continuous torque)
    without latency spikes from blocking intervals or melting the substrate.
    """
    hm = HardwareMonitor()
    # Hard thermal runaway veto
    if hm.thermal_runaway:
        return False

    # Depleted cooling margin veto (< 5% cooling headroom remaining)
    if hm.p_waste > 0.0 and hm.thermal_margin < 0.05:
        return False

    try:
        cpu_load = psutil.cpu_percent(interval=None)
        mem_info = psutil.virtual_memory()
        free_ram_gb = mem_info.available / (1024 ** 3)
        
        # Under thermal stress (margin < 25%), apply strict thermal throttling
        max_cpu = 65.0 if (hm.p_waste > 0.0 and hm.thermal_margin < 0.25) else 90.0
        return cpu_load < max_cpu and free_ram_gb > 0.5
    except Exception:
        return True

def get_thermal_affordance() -> dict:
    """Returns current thermodynamic telemetry and compute affordance status."""
    hm = HardwareMonitor()
    return {
        "thermal_runaway": hm.thermal_runaway,
        "p_waste": hm.p_waste,
        "p_cool_max": hm.p_cool_max,
        "thermal_margin": hm.thermal_margin,
        "eta_stack": hm.eta_stack,
        "thermal_source": hm.thermal_source,
        "has_compute_affordance": has_compute_affordance()
    }

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
