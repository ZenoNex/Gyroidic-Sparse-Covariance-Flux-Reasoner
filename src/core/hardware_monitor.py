import psutil
import os
import math
from enum import Enum
from typing import Dict, Any, Optional, Union

class ComputeAffordanceType(str, Enum):
    """
    Categorical typing for compute affordances to prevent scalarization collapse.
    Each computational regime maintains its own gauge, constraints, and operational bounds.
    """
    SILICON = "silicon"             # Host CPU and RAM physical capacity
    THERMAL = "thermal"             # Thermodynamic waste heat vs cooling (Carnot-Möbius)
    GPU_SOVEREIGN = "gpu_sovereign" # OpenCL Silicon Sovereignty Engine / Queue B
    TOPOLOGICAL = "topological"     # CALM abort score, geometric manifold stability
    P2P_MESH = "p2p_mesh"           # Bonfire nomadic ring peer budget
    DIEGETIC = "diegetic"           # Simulation kinematics, conversational pressure
    ALGORITHMIC = "algorithmic"     # Heavy second-order checks (ForbiddenSmoothing, etc.)

    @classmethod
    def from_str(cls, val: Union[str, 'ComputeAffordanceType']) -> 'ComputeAffordanceType':
        if isinstance(val, cls):
            return val
        s = str(val).lower().strip()
        for member in cls:
            if member.value == s or member.name.lower() == s:
                return member
        aliases = {
            "hardware": cls.SILICON,
            "cpu": cls.SILICON,
            "carnot": cls.THERMAL,
            "heat": cls.THERMAL,
            "opencl": cls.GPU_SOVEREIGN,
            "gpu": cls.GPU_SOVEREIGN,
            "sovereign": cls.GPU_SOVEREIGN,
            "calm": cls.TOPOLOGICAL,
            "geometry": cls.TOPOLOGICAL,
            "p2p": cls.P2P_MESH,
            "mesh": cls.P2P_MESH,
            "collective": cls.P2P_MESH,
            "simulation": cls.DIEGETIC,
            "fauna": cls.DIEGETIC,
            "smoothing": cls.ALGORITHMIC,
            "second_order": cls.ALGORITHMIC,
        }
        return aliases.get(s, cls.SILICON)


class TypedComputeAffordance:
    """
    Individually typed compute affordance token.
    Enforces domain isolation between different computational regimes (Silicon, Thermal,
    GPU Sovereign, Topological, P2P Mesh, Diegetic, Algorithmic) and provides mathematically
    grounded gauge transformations between them.
    """
    def __init__(
        self,
        affordance_type: ComputeAffordanceType,
        gauge: float,
        is_admissible: bool,
        telemetry: Optional[Dict[str, Any]] = None,
        source: str = "hardware_monitor"
    ):
        self.affordance_type = affordance_type
        self.gauge = max(0.0, min(1.0, float(gauge)))
        self.is_admissible = bool(is_admissible)
        self.telemetry = telemetry or {}
        self.source = source

    def __bool__(self) -> bool:
        """Admissibility predicate: True if admissible and non-zero gauge."""
        return self.is_admissible and (self.gauge > 0.0)

    def item(self) -> float:
        """Returns the scalar gauge magnitude in [0, 1]."""
        return self.gauge

    def __repr__(self) -> str:
        status = "ADMISSIBLE" if self.is_admissible else "VETOED"
        return f"TypedComputeAffordance(type='{self.affordance_type.value}', gauge={self.gauge:.3f}, status={status})"

    def gauge_transform(
        self,
        target_type: Union[str, ComputeAffordanceType],
        coupling: float = 1.0
    ) -> 'TypedComputeAffordance':
        """
        Calculates a gauge transformation to another affordance type.
        Applies a transfer coupling factor without merging domain identities.
        """
        tgt = ComputeAffordanceType.from_str(target_type)
        transformed_gauge = max(0.0, min(1.0, self.gauge * float(coupling)))
        return TypedComputeAffordance(
            affordance_type=tgt,
            gauge=transformed_gauge,
            is_admissible=self.is_admissible and (transformed_gauge > 0.05),
            telemetry={**self.telemetry, "origin_type": self.affordance_type.value, "coupling": coupling},
            source=f"gauge_transform({self.affordance_type.value}->{tgt.value})"
        )


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

            # Collective Mesh Signals
            cls._instance.p2p_mesh_gauge = 0.8
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'HardwareMonitor':
        if cls._instance is None:
            cls()
        return cls._instance

    def update_topology(self, abort_score: float, gauge: float):
        """
        Ingest continuous topological signals from the CALM predictor.
        Allows the hardware monitor to gate PyOpenCL scaling on geometry, not just silicon load.
        """
        self.topological_abort_score = float(abort_score)
        self.topological_gauge = float(gauge)

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

    def update_p2p_mesh(self, mesh_gauge: float):
        """Update collective nomadic ring mesh budget."""
        self.p2p_mesh_gauge = max(0.0, min(1.0, float(mesh_gauge)))

    def get_typed_affordance(
        self,
        affordance_type: Union[str, ComputeAffordanceType] = ComputeAffordanceType.SILICON
    ) -> TypedComputeAffordance:
        """
        Evaluates the specific typed compute affordance and gauge for the requested domain.
        """
        atype = ComputeAffordanceType.from_str(affordance_type)
        
        # 1. Silicon gauge: host CPU and RAM
        try:
            cpu_load = psutil.cpu_percent(interval=None)
            mem_info = psutil.virtual_memory()
            free_ram_gb = mem_info.available / (1024 ** 3)
            cpu_headroom = max(0.0, (100.0 - cpu_load) / 100.0)
            ram_headroom = min(1.0, free_ram_gb / 4.0)
            silicon_gauge = max(0.0, min(1.0, 0.6 * cpu_headroom + 0.4 * ram_headroom))
            max_cpu = 65.0 if (self.p_waste > 0.0 and self.thermal_margin < 0.25) else 90.0
            silicon_admissible = (cpu_load < max_cpu) and (free_ram_gb > 0.5)
        except Exception:
            cpu_load = 50.0
            free_ram_gb = 4.0
            silicon_gauge = 0.5
            silicon_admissible = True

        # 2. Thermal gauge: Carnot cooling capacity and waste heat
        if self.thermal_runaway:
            thermal_gauge = 0.0
            thermal_admissible = False
        elif self.p_waste > 0.0:
            thermal_gauge = max(0.0, min(1.0, self.thermal_margin))
            thermal_admissible = self.thermal_margin >= 0.05
        else:
            thermal_gauge = 1.0
            thermal_admissible = True

        if atype == ComputeAffordanceType.SILICON:
            effective_gauge = silicon_gauge * max(0.1, thermal_gauge)
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=effective_gauge,
                is_admissible=silicon_admissible and not self.thermal_runaway,
                telemetry={"cpu_load": cpu_load, "free_ram_gb": free_ram_gb, "raw_gauge": silicon_gauge}
            )

        elif atype == ComputeAffordanceType.THERMAL:
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=thermal_gauge,
                is_admissible=thermal_admissible,
                telemetry={
                    "p_waste": self.p_waste,
                    "p_cool_max": self.p_cool_max,
                    "thermal_margin": self.thermal_margin,
                    "thermal_runaway": self.thermal_runaway,
                    "source": self.thermal_source
                }
            )

        elif atype == ComputeAffordanceType.GPU_SOVEREIGN:
            # OpenCL Queue B: coupled to Silicon, OpenCL, and Thermal backpressure
            if self.thermal_runaway or thermal_gauge < 0.10:
                gpu_gauge = 0.0
                gpu_admissible = False
            else:
                gpu_gauge = silicon_gauge * thermal_gauge
                gpu_admissible = silicon_admissible and (thermal_gauge >= 0.15)
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=gpu_gauge,
                is_admissible=gpu_admissible,
                telemetry={
                    "silicon_coupling": silicon_gauge,
                    "thermal_coupling": thermal_gauge,
                    "thermal_runaway": self.thermal_runaway
                }
            )

        elif atype == ComputeAffordanceType.TOPOLOGICAL:
            # CALM predictor: abort score < 0.85 allows standard optimization
            abort = self.topological_abort_score
            calm_g = self.topological_gauge
            topo_gauge = max(0.0, min(1.0, (1.0 - abort) * 0.7 + calm_g * 0.3))
            topo_admissible = (abort < 0.85) and not self.thermal_runaway
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=topo_gauge,
                is_admissible=topo_admissible,
                telemetry={"abort_score": abort, "topological_gauge": calm_g}
            )

        elif atype == ComputeAffordanceType.P2P_MESH:
            mesh_gauge = self.p2p_mesh_gauge
            mesh_admissible = (mesh_gauge > 0.10) and not self.thermal_runaway
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=mesh_gauge,
                is_admissible=mesh_admissible,
                telemetry={"p2p_mesh_gauge": mesh_gauge}
            )

        elif atype == ComputeAffordanceType.DIEGETIC:
            diegetic_gauge = silicon_gauge * max(0.2, thermal_gauge)
            diegetic_admissible = silicon_admissible and not self.thermal_runaway
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=diegetic_gauge,
                is_admissible=diegetic_admissible,
                telemetry={"diegetic_gauge": diegetic_gauge}
            )

        elif atype == ComputeAffordanceType.ALGORITHMIC:
            abort = self.topological_abort_score
            algo_gauge = silicon_gauge * thermal_gauge * max(0.1, 1.0 - abort)
            algo_admissible = silicon_admissible and thermal_admissible and (abort < 0.70) and not self.thermal_runaway
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=algo_gauge,
                is_admissible=algo_admissible,
                telemetry={"algo_gauge": algo_gauge, "silicon_gauge": silicon_gauge, "thermal_gauge": thermal_gauge}
            )

        return TypedComputeAffordance(affordance_type=atype, gauge=0.5, is_admissible=True)

    def gauge_between(
        self,
        source_type: Union[str, ComputeAffordanceType],
        target_type: Union[str, ComputeAffordanceType]
    ) -> float:
        """
        Computes the cross-type gauge transfer coupling between source and target affordances.
        Evaluates how headroom or backpressure in one domain transforms into available budget
        in another domain without naive scalarization.
        """
        src = ComputeAffordanceType.from_str(source_type)
        tgt = ComputeAffordanceType.from_str(target_type)
        if src == tgt:
            return 1.0

        src_aff = self.get_typed_affordance(src)
        tgt_aff = self.get_typed_affordance(tgt)

        if src == ComputeAffordanceType.THERMAL:
            if tgt == ComputeAffordanceType.GPU_SOVEREIGN:
                return float(src_aff.gauge if not self.thermal_runaway else 0.0)
            elif tgt == ComputeAffordanceType.ALGORITHMIC:
                return float((src_aff.gauge ** 2) if not self.thermal_runaway else 0.0)
            elif tgt == ComputeAffordanceType.SILICON:
                return float(max(0.5, src_aff.gauge) if not self.thermal_runaway else 0.0)
            else:
                return float(max(0.2, src_aff.gauge) if not self.thermal_runaway else 0.0)

        elif src == ComputeAffordanceType.TOPOLOGICAL:
            abort = self.topological_abort_score
            if tgt == ComputeAffordanceType.ALGORITHMIC:
                return float(max(0.0, 1.0 - abort))
            elif tgt == ComputeAffordanceType.GPU_SOVEREIGN:
                return 1.0 if abort > 0.8 else float(tgt_aff.gauge)

        elif src == ComputeAffordanceType.P2P_MESH and tgt == ComputeAffordanceType.SILICON:
            return float(min(src_aff.gauge, tgt_aff.gauge))

        return float(math.sqrt(src_aff.gauge * tgt_aff.gauge))

    def get_affordance_gauges(self) -> Dict[str, float]:
        """Returns the dictionary of all typed affordance gauge values in [0, 1]."""
        return {
            atype.value: self.get_typed_affordance(atype).gauge
            for atype in ComputeAffordanceType
        }

    def has_headroom(self) -> bool:
        """
        Evaluates physical headroom (CPU/RAM) in conjunction with topological stability (CALM)
        and thermodynamic ledger state (Carnot-Möbius).
        - If in thermal runaway, suppress computation to prevent melting/rupture.
        - If geometry is collapsing (abort > 0.8), authorize PyOpenCL TailSlayer to save manifold
          unless silicon is exhausted (> 95% CPU).
        - If geometry is calm (abort < 0.2), suppress PyOpenCL to conserve logic cycles.
        """
        if self.thermal_runaway:
            return False

        try:
            cpu_load = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            free_ram_gb = mem_info.available / (1024 ** 3)
            
            if cpu_load == 0.0:
                self.desync_anomalies += 1
            else:
                self.desync_anomalies = max(0, self.desync_anomalies - 1)

            if self.desync_anomalies > 5:
                return False

            if self.topological_abort_score > 0.8:
                return cpu_load < 95.0 and free_ram_gb > 1.0
                
            if self.topological_abort_score < 0.2:
                return False
                
            effective_cpu_threshold = self.cpu_threshold * max(0.5, self.thermal_margin)
            return cpu_load < effective_cpu_threshold and free_ram_gb > self.ram_threshold_gb
        except Exception:
            return False


def has_headroom() -> bool:
    """Helper function to get headroom status from the centralized monitor."""
    return HardwareMonitor.get_instance().has_headroom()

def has_compute_affordance(
    affordance_type: Optional[Union[str, ComputeAffordanceType]] = None
) -> bool:
    """
    Evaluates computational affordance.
    If affordance_type is provided, evaluates the specific typed affordance.
    If None, evaluates the composite affordance (Silicon AND Thermal).
    """
    hm = HardwareMonitor.get_instance()
    if affordance_type is not None:
        return bool(hm.get_typed_affordance(affordance_type))
    
    silicon = hm.get_typed_affordance(ComputeAffordanceType.SILICON)
    thermal = hm.get_typed_affordance(ComputeAffordanceType.THERMAL)
    return bool(silicon and thermal)

def get_typed_affordance(
    affordance_type: Union[str, ComputeAffordanceType] = ComputeAffordanceType.SILICON
) -> TypedComputeAffordance:
    """Helper function to retrieve a specific typed affordance token."""
    return HardwareMonitor.get_instance().get_typed_affordance(affordance_type)

def get_affordance_gauges() -> Dict[str, float]:
    """Helper function to get all affordance gauges in [0, 1]."""
    return HardwareMonitor.get_instance().get_affordance_gauges()

def gauge_between(
    source_type: Union[str, ComputeAffordanceType],
    target_type: Union[str, ComputeAffordanceType]
) -> float:
    """Helper function to compute cross-type gauge transfer coupling."""
    return HardwareMonitor.get_instance().gauge_between(source_type, target_type)

def get_thermal_affordance() -> dict:
    """Returns current thermodynamic telemetry, typed gauges, and affordance status."""
    hm = HardwareMonitor.get_instance()
    return {
        "thermal_runaway": hm.thermal_runaway,
        "p_waste": hm.p_waste,
        "p_cool_max": hm.p_cool_max,
        "thermal_margin": hm.thermal_margin,
        "eta_stack": hm.eta_stack,
        "thermal_source": hm.thermal_source,
        "has_compute_affordance": has_compute_affordance(),
        "gauges": hm.get_affordance_gauges()
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
        
        ram_speed_mhz = 2400
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
                
        speed_multiplier = ram_speed_mhz / 2400.0
        estimated_nodes = int((available_mb / 0.8) * speed_multiplier)
        admissibility_boost = 1.5 
        dimensional_limit = int(estimated_nodes * admissibility_boost)
        return max(50, dimensional_limit)
        
    except Exception as e:
        print(f"[TAHI] Headroom computation degraded: {e}")
        return 50000
