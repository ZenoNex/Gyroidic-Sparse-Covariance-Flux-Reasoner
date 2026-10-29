import psutil
import os
import math
import time
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
    COMPUTE_BATCH = "compute_batch" # Matrioshka Mandelbulb partitioning and situational batch capacity

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
            "batch": cls.COMPUTE_BATCH,
            "batching": cls.COMPUTE_BATCH,
            "matrioshka": cls.COMPUTE_BATCH,
            "mandelbulb": cls.COMPUTE_BATCH,
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
        p_waste: float = 0.0,
        p_cool_max: float = 1000.0,
        thermal_runaway: bool = False,
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
        self.thermal_margin = max(0.0, (self.p_cool_max - self.p_waste) / self.p_cool_max)
        # Hysteretic cooling recovery: if waste heat drops below 75% cooling capacity and margin > 25%, clear thermal runaway
        if self.thermal_runaway and self.p_waste <= self.p_cool_max * 0.75 and self.thermal_margin > 0.25 and not thermal_runaway:
            self.thermal_runaway = False
        else:
            self.thermal_runaway = bool(thermal_runaway) or (self.p_waste > self.p_cool_max)
        self.eta_stack = float(eta_stack)
        self.thermal_source = str(source)

    def reset_thermodynamics(self):
        """Manually reset thermal runaway latch and restore full compute affordance."""
        self.thermal_runaway = False
        self.p_waste = 0.0
        self.thermal_margin = 1.0

    def get_cpu_gpu_layout(self) -> Dict[str, Any]:
        """
        Inspects physical and logical CPU cores, per-core utilization, frequency layouts,
        and PyOpenCL GPU/accelerator compute core layouts to recognize different hardware
        topologies (e.g. heterogeneous P/E core clusters, GPU compute unit scales).
        """
        layout = {
            "cpu": {
                "physical_cores": 1,
                "logical_cores": 1,
                "per_core_percent": [],
                "frequencies_mhz": {},
                "is_heterogeneous": False,
                "performance_cores": [],
                "efficiency_cores": []
            },
            "gpu": {
                "available": False,
                "devices": []
            }
        }
        
        # 1. CPU Layout via psutil
        try:
            phys = psutil.cpu_count(logical=False) or 1
            logi = psutil.cpu_count(logical=True) or phys
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            layout["cpu"]["physical_cores"] = phys
            layout["cpu"]["logical_cores"] = logi
            layout["cpu"]["per_core_percent"] = per_cpu
            
            try:
                freqs = psutil.cpu_freq(percpu=True)
                if freqs:
                    layout["cpu"]["frequencies_mhz"] = {
                        f"core_{i}": {"current": f.current, "min": f.min, "max": f.max}
                        for i, f in enumerate(freqs)
                    }
                    max_freqs = [f.max for f in freqs if f.max > 0]
                    if len(set(max_freqs)) > 1:
                        layout["cpu"]["is_heterogeneous"] = True
                        median_freq = sorted(max_freqs)[len(max_freqs) // 2]
                        for i, f in enumerate(freqs):
                            if f.max >= median_freq:
                                layout["cpu"]["performance_cores"].append(i)
                            else:
                                layout["cpu"]["efficiency_cores"].append(i)
                else:
                    global_freq = psutil.cpu_freq()
                    if global_freq:
                        layout["cpu"]["frequencies_mhz"] = {
                            "global": {"current": global_freq.current, "min": global_freq.min, "max": global_freq.max}
                        }
            except Exception:
                pass
        except Exception as e:
            layout["cpu"]["error"] = str(e)

        # 2. GPU Core Layout via PyOpenCL
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            devices_info = []
            for plat in platforms:
                for dev in plat.get_devices():
                    dev_type_str = "GPU" if (dev.type & cl.device_type.GPU) else ("CPU" if (dev.type & cl.device_type.CPU) else "ACCELERATOR")
                    is_disc = False
                    dev_name_lower = dev.name.lower()
                    if dev_type_str == "GPU":
                        if any(k in dev_name_lower for k in ["nvidia", "geforce", "gtx", "rtx", "radeon", "discrete"]):
                            is_disc = True
                    
                    dev_meta = {
                        "name": dev.name.strip(),
                        "platform": plat.name.strip(),
                        "type": dev_type_str,
                        "compute_units": getattr(dev, "max_compute_units", 1),
                        "max_work_group_size": getattr(dev, "max_work_group_size", 1),
                        "max_work_item_sizes": list(getattr(dev, "max_work_item_sizes", [1])),
                        "global_mem_mb": getattr(dev, "global_mem_size", 0) / (1024 * 1024),
                        "local_mem_kb": getattr(dev, "local_mem_size", 0) / 1024,
                        "is_discrete": is_disc
                    }
                    devices_info.append(dev_meta)
            if devices_info:
                layout["gpu"]["available"] = True
                layout["gpu"]["devices"] = devices_info
        except Exception:
            layout["gpu"]["available"] = False

        return layout

    get_hardware_core_layout = get_cpu_gpu_layout

    def update_p2p_mesh(
        self,
        mesh_gauge: float,
        freenet_online: Optional[bool] = None,
        active_peers: int = 0,
        ghost_latency_ms: Optional[float] = None,
        kelly_consensus_headroom: Optional[float] = None,
    ):
        """Update collective nomadic ring mesh budget with live Freenet, Ghost latency, and peer telemetry."""
        base_gauge = max(0.0, min(1.0, float(mesh_gauge)))
        if freenet_online is not None:
            self.freenet_online = bool(freenet_online)
            if freenet_online:
                base_gauge = max(base_gauge, 0.70)
        if active_peers > 0:
            self.active_mesh_peers = int(active_peers)
            base_gauge = min(1.0, max(base_gauge, 0.50 + 0.10 * active_peers))
        if ghost_latency_ms is not None:
            self.ghost_latency_ms = max(0.0, float(ghost_latency_ms))
            # Attenuate mesh gauge only under extreme topological latency (> 1000 ms)
            latency_factor = max(0.25, 1.0 - min(0.75, self.ghost_latency_ms / 2000.0))
            base_gauge = min(1.0, base_gauge * latency_factor)
        if kelly_consensus_headroom is not None:
            self.kelly_consensus_headroom = max(0.0, min(1.0, float(kelly_consensus_headroom)))
            base_gauge = min(1.0, 0.75 * base_gauge + 0.25 * self.kelly_consensus_headroom)
        self.p2p_mesh_gauge = base_gauge

    def get_typed_affordance(
        self,
        affordance_type: Union[str, ComputeAffordanceType] = ComputeAffordanceType.SILICON
    ) -> TypedComputeAffordance:
        """
        Evaluates the specific typed compute affordance and gauge for the requested domain.
        """
        atype = ComputeAffordanceType.from_str(affordance_type)
        core_layout = self.get_cpu_gpu_layout()
        
        # 1. Silicon gauge: host CPU and RAM
        try:
            cpu_load = psutil.cpu_percent(interval=None)
            mem_info = psutil.virtual_memory()
            free_ram_gb = mem_info.available / (1024 ** 3)
            cpu_headroom = max(0.0, (100.0 - cpu_load) / 100.0)
            ram_headroom = min(1.0, free_ram_gb / 4.0)
            silicon_gauge = max(0.1, min(1.0, 0.6 * cpu_headroom + 0.4 * ram_headroom))
            max_cpu = 65.0 if (self.p_waste > 0.0 and self.thermal_margin < 0.25) else 95.0
            silicon_admissible = (cpu_load < max_cpu or free_ram_gb > 1.5) and (free_ram_gb > 0.5) and not self.thermal_runaway
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
                telemetry={
                    "cpu_load": cpu_load,
                    "free_ram_gb": free_ram_gb,
                    "raw_gauge": silicon_gauge,
                    "core_layout": core_layout["cpu"]
                }
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
                    "thermal_runaway": self.thermal_runaway,
                    "gpu_devices": core_layout["gpu"]["devices"]
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
                telemetry={
                    "p2p_mesh_gauge": mesh_gauge,
                    "freenet_online": getattr(self, 'freenet_online', False),
                    "active_peers": getattr(self, 'active_mesh_peers', 0),
                    "ghost_latency_ms": getattr(self, 'ghost_latency_ms', 0.0),
                    "kelly_consensus_headroom": getattr(self, 'kelly_consensus_headroom', 0.5),
                }
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

        elif atype == ComputeAffordanceType.COMPUTE_BATCH:
            batch_gauge = silicon_gauge * max(0.2, thermal_gauge)
            batch_admissible = silicon_admissible and not self.thermal_runaway and (free_ram_gb > 0.5)
            return TypedComputeAffordance(
                affordance_type=atype,
                gauge=batch_gauge,
                is_admissible=batch_admissible,
                telemetry={
                    "silicon_gauge": silicon_gauge,
                    "free_ram_gb": free_ram_gb,
                    "max_batch_size": max(1, int(batch_gauge * 64)),
                    "max_matrioshka_shells": 5 if batch_gauge > 0.4 else (3 if batch_gauge > 0.2 else 1)
                }
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

    def has_compute_affordance(
        self,
        affordance_type: Optional[Union[str, ComputeAffordanceType]] = None
    ) -> bool:
        """
        Evaluates computational affordance for the specified domain.
        If affordance_type is provided, checks specific typed affordance admissibility.
        If None, evaluates composite affordance (Silicon AND Thermal).
        """
        if affordance_type is not None:
            return bool(self.get_typed_affordance(affordance_type))
        silicon = self.get_typed_affordance(ComputeAffordanceType.SILICON)
        thermal = self.get_typed_affordance(ComputeAffordanceType.THERMAL)
        return bool(silicon and thermal)

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
            cpu_load = psutil.cpu_percent(interval=None)
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
                
            effective_cpu_threshold = self.cpu_threshold * max(0.5, self.thermal_margin)
            return cpu_load < effective_cpu_threshold and free_ram_gb > self.ram_threshold_gb
        except Exception:
            return False

    def evaluate_matrioshka_mandelbulb_affordance(
        self,
        task_complexity: float = 1.0,
        dt: float = 1.0,
        demand_dim: int = 32
    ) -> Dict[str, Any]:
        """
        Explicitly positions whether the system can afford Matrioshka Mandelbulb
        partitioning of compute batching for high-value tasks.
        
        Evaluates Silicon headroom, Carnot-Möbius thermal dissipation, and derivative
        time (dt) elasticity to prevent false-positive Leontief vetoes and self-slashing.
        """
        silicon_aff = self.get_typed_affordance(ComputeAffordanceType.SILICON)
        thermal_aff = self.get_typed_affordance(ComputeAffordanceType.THERMAL)
        batch_aff = self.get_typed_affordance(ComputeAffordanceType.COMPUTE_BATCH)
        gpu_aff = self.get_typed_affordance(ComputeAffordanceType.GPU_SOVEREIGN)

        # Baseline affordability: non-thermal-runaway with minimum viable silicon headroom
        can_afford = (not self.thermal_runaway) and (silicon_aff.gauge > 0.08)
        
        # Determine optimal Matrioshka shell depth
        # Higher complexity or lower immediate slice capacity -> deeper Matrioshka partitioning
        effective_budget = max(0.15, silicon_aff.gauge * max(0.5, thermal_aff.gauge))
        if task_complexity > 2.0 or effective_budget < 0.4:
            optimal_shells = 5
        elif task_complexity > 1.2 or effective_budget < 0.6:
            optimal_shells = 3
        else:
            optimal_shells = 2

        # Temporal amortization factor: effective_time_steps = shells * max(1, 1 / dt)
        clamped_dt = max(0.01, min(2.0, float(dt)))
        effective_time_steps = max(1.0, float(optimal_shells) * max(1.0, 1.0 / clamped_dt))
        amortization_factor = effective_time_steps

        # Situational batch capacity (max batch size that fits in current silicon/RAM headroom)
        try:
            mem_info = psutil.virtual_memory()
            free_gb = mem_info.available / (1024 ** 3)
        except Exception:
            free_gb = 4.0
        batch_capacity = max(1, min(128, int(free_gb * 16 * effective_budget)))

        # Recommended Mandelbulb fractal iteration power based on thermal margin
        if self.thermal_margin > 0.7 and bool(gpu_aff):
            mandelbulb_power = 8
        elif self.thermal_margin > 0.4:
            mandelbulb_power = 6
        else:
            mandelbulb_power = 4

        return {
            "can_afford_partitioning": can_afford,
            "optimal_matrioshka_shells": optimal_shells,
            "batch_capacity": batch_capacity,
            "effective_time_steps": effective_time_steps,
            "amortization_factor": amortization_factor,
            "mandelbulb_power_budget": mandelbulb_power,
            "clamped_dt": clamped_dt,
            "effective_budget": effective_budget,
            "silicon_gauge": silicon_aff.gauge,
            "thermal_gauge": thermal_aff.gauge,
            "thermal_runaway": self.thermal_runaway,
            "is_high_value_exempt": True
        }

    def probe_compute_feelers(self) -> Dict[str, Any]:
        """
        Extends local and community compute affordance feelers across:
          1. Local Silicon & DRAM t_RFC stall latency feeler (nanosecond jitter probe)
          2. Local GPU Sovereign Queue A / Queue B TailSlayer first-to-finish headroom
          3. Community P2P Mesh (BonfireNomadicRing / FreenetGhostCaller) federation headroom
          4. Directional gauge coupling between Local Silicon and Community P2P Mesh
        """
        import time
        t0_ns = time.perf_counter_ns()
        # Non-allocating L1/DRAM cache-line stride touch to sense t_RFC refresh stall
        _accum = 0
        for i in range(64):
            _accum ^= (i * 2654435761) & 0xFFFFFFFF
        t_rfc_feeler_ns = max(1, time.perf_counter_ns() - t0_ns)
        # Typical unstalled loop is < 50,000 ns; DRAM t_RFC / OS scheduling stall spikes higher
        t_rfc_stall_detected = bool(t_rfc_feeler_ns > 150_000)

        silicon_aff = self.get_typed_affordance(ComputeAffordanceType.SILICON)
        thermal_aff = self.get_typed_affordance(ComputeAffordanceType.THERMAL)
        gpu_aff = self.get_typed_affordance(ComputeAffordanceType.GPU_SOVEREIGN)
        topo_aff = self.get_typed_affordance(ComputeAffordanceType.TOPOLOGICAL)
        mesh_aff = self.get_typed_affordance(ComputeAffordanceType.P2P_MESH)

        local_to_community_gauge = self.gauge_between(
            ComputeAffordanceType.SILICON, ComputeAffordanceType.P2P_MESH
        )
        community_to_local_gauge = self.gauge_between(
            ComputeAffordanceType.P2P_MESH, ComputeAffordanceType.SILICON
        )

        # Community offload advantage: when P2P mesh headroom exceeds constrained local silicon/thermal
        community_offload_recommended = bool(
            bool(mesh_aff)
            and getattr(self, "active_mesh_peers", 0) > 0
            and (mesh_aff.gauge > silicon_aff.gauge * 1.15 or thermal_aff.gauge < 0.35)
        )

        return {
            "t_rfc_feeler_ns": int(t_rfc_feeler_ns),
            "t_rfc_stall_detected": t_rfc_stall_detected,
            "local_silicon_gauge": float(silicon_aff.gauge),
            "local_thermal_gauge": float(thermal_aff.gauge),
            "local_gpu_sovereign_gauge": float(gpu_aff.gauge),
            "topological_gauge": float(topo_aff.gauge),
            "community_p2p_mesh_gauge": float(mesh_aff.gauge),
            "freenet_online": bool(getattr(self, "freenet_online", False)),
            "active_mesh_peers": int(getattr(self, "active_mesh_peers", 0)),
            "ghost_latency_ms": float(getattr(self, "ghost_latency_ms", 0.0)),
            "kelly_consensus_headroom": float(getattr(self, "kelly_consensus_headroom", 0.5)),
            "local_to_community_gauge": float(local_to_community_gauge),
            "community_to_local_gauge": float(community_to_local_gauge),
            "community_offload_recommended": community_offload_recommended,
        }

    def evaluate_speculative_routing_timings(
        self,
        execution_ms: float = 0.0,
        h_spec: float = 0.5,
        pas_h: float = 0.75,
        epsilon_fastpass: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evaluates speculative routing times, thermal latency guards, and connective routing paths:
          - 500ms "Restless" Thermal Latency Guard (SIC-FA-ADMM block limit -> clFinish / Lazarus throttle)
          - Refresh-Triggered Play (t_RFC memory stall -> RP^4 Projective Routing / Lazarus Transition)
          - Spectral Speculation Fastpass (H_spec < epsilon -> 90% ADMM bypass)
          - 7-Step Speculative Chebyshev Trajectory Horizon (K_spec in [1, 7] bounded by Catalan C_{K-1})
          - Local TailSlayer Dual-Queue (Queue A + Queue B first-to-finish) vs Community Bonfire Federation
        """
        feelers = self.probe_compute_feelers()
        exceeds_500ms_guard = bool(float(execution_ms) > 500.0)
        t_rfc_stall = bool(feelers["t_rfc_stall_detected"])
        is_spectral_fastpass = bool(float(h_spec) < float(epsilon_fastpass) and not exceeds_500ms_guard)

        # Compute adaptive speculative lookahead horizon K_spec in [1, 7]
        # High PAS_h and high silicon/GPU gauge unlock full 7-step Chebyshev trajectory draft
        composite_headroom = 0.5 * feelers["local_silicon_gauge"] + 0.5 * feelers["local_thermal_gauge"]
        if exceeds_500ms_guard or self.thermal_runaway:
            k_spec = 1
        elif pas_h >= 0.65 and composite_headroom >= 0.45:
            k_spec = 7
        elif pas_h >= 0.45 and composite_headroom >= 0.25:
            k_spec = 4
        else:
            k_spec = 2

        # Catalan number C_{k_spec - 1} bounds the Stasheff Associahedron parenthesizations of the draft
        n_cat = max(0, k_spec - 1)
        catalan_branch_bound = math.comb(2 * n_cat, n_cat) // (n_cat + 1)

        # Determine connective speculative routing mode
        if exceeds_500ms_guard or t_rfc_stall:
            # Turn hardware stall / 500ms thermal guard into generative RP^4 Void Lazarus Play
            routing_mode = "rp4_void_lazarus_play"
            recommended_dt_scale = 1.35  # Play dilation while hardware finishes/cools
        elif is_spectral_fastpass:
            routing_mode = "spectral_fastpass"
            recommended_dt_scale = 1.15
        elif feelers["community_offload_recommended"]:
            routing_mode = "bonfire_community_federation"
            recommended_dt_scale = 1.0
        elif feelers["local_gpu_sovereign_gauge"] >= 0.25:
            routing_mode = "tailslayer_dual_queue"
            recommended_dt_scale = 1.0
        else:
            routing_mode = "scccg_gravity_well"
            recommended_dt_scale = 0.75  # Seriousness contraction

        return {
            "routing_mode": routing_mode,
            "is_spectral_fastpass": is_spectral_fastpass,
            "exceeds_500ms_thermal_guard": exceeds_500ms_guard,
            "t_rfc_stall_detected": t_rfc_stall,
            "t_rfc_feeler_ns": feelers["t_rfc_feeler_ns"],
            "execution_ms": float(execution_ms),
            "speculative_horizon_steps": int(k_spec),
            "catalan_associahedron_bound": int(catalan_branch_bound),
            "recommended_dt_scale": float(recommended_dt_scale),
            "community_offload_recommended": feelers["community_offload_recommended"],
            "feelers": feelers,
        }

    def enforce_500ms_restless_thermal_guard(
        self,
        execution_ms: float,
        silicon_engine: Optional[Any] = None,
        cooldown_sleep_sec: float = 0.05,
        pas_h: float = 0.75,
        h_spec: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Enforces the 500ms "Restless" Thermal Latency Guard (Instruction 4,
        tailslayer_pyopencl_mempalace_BigGan_inspirationdoc.md & TAILSLAYER_PYOPENCL_ARCHITECTURE.md):
          - If a step or SIC-FA-ADMM block exceeds 500ms, calls clFinish() on active
            OpenCL command queues (Queue A / Queue B) to drain pending hardware work.
          - Updates Carnot-Mobius thermodynamic waste heat telemetry.
          - Applies a micro-sleep cooldown (`time.sleep(cooldown_sleep_sec)`) to prevent
            CPU/GPU thermal runaway and host lockup on 4GB VRAM / 8GB RAM laptops.
          - Evaluates speculative routing timings to transition into RP^4 Void Lazarus Play.
        """
        exceeded = float(execution_ms) > 500.0
        cl_finished = False

        if exceeded:
            # 1. Drain PyOpenCL queues via clFinish() if silicon_engine is present
            if silicon_engine is not None:
                for q_attr in ("queue", "queue_a", "queue_b"):
                    q_obj = getattr(silicon_engine, q_attr, None)
                    if q_obj is not None and hasattr(q_obj, "finish"):
                        try:
                            q_obj.finish()
                            cl_finished = True
                        except Exception:
                            pass

            # 2. Register transient thermal load spike without permanently latching runaway
            inferred_waste = min(self.p_cool_max * 0.92, max(self.p_waste, (float(execution_ms) / 500.0) * 450.0))
            self.update_thermodynamics(
                p_waste=inferred_waste,
                p_cool_max=self.p_cool_max,
                thermal_runaway=False,
                eta_stack=self.eta_stack,
                source="restless_500ms_thermal_guard",
            )

            # 3. Apply non-blocking micro-cooldown sleep to protect host node
            if cooldown_sleep_sec > 0.0:
                time.sleep(min(0.25, float(cooldown_sleep_sec)))

        routing = self.evaluate_speculative_routing_timings(
            execution_ms=execution_ms,
            h_spec=h_spec,
            pas_h=pas_h,
        )
        routing["cl_finish_invoked"] = cl_finished
        routing["cooldown_applied_sec"] = float(cooldown_sleep_sec) if exceeded else 0.0
        return routing


def enforce_500ms_restless_thermal_guard(
    execution_ms: float,
    silicon_engine: Optional[Any] = None,
    cooldown_sleep_sec: float = 0.05,
    pas_h: float = 0.75,
    h_spec: float = 0.5,
) -> Dict[str, Any]:
    """Module-level helper for the 500ms Restless Thermal Latency Guard."""
    return HardwareMonitor.get_instance().enforce_500ms_restless_thermal_guard(
        execution_ms=execution_ms,
        silicon_engine=silicon_engine,
        cooldown_sleep_sec=cooldown_sleep_sec,
        pas_h=pas_h,
        h_spec=h_spec,
    )


def probe_compute_feelers() -> Dict[str, Any]:
    """Module-level helper to probe local and community compute affordance feelers."""
    return HardwareMonitor.get_instance().probe_compute_feelers()


def evaluate_speculative_routing_timings(
    execution_ms: float = 0.0,
    h_spec: float = 0.5,
    pas_h: float = 0.75,
    epsilon_fastpass: float = 0.25,
) -> Dict[str, Any]:
    """Module-level helper to evaluate speculative routing times, guards, and connective paths."""
    return HardwareMonitor.get_instance().evaluate_speculative_routing_timings(
        execution_ms=execution_ms,
        h_spec=h_spec,
        pas_h=pas_h,
        epsilon_fastpass=epsilon_fastpass,
    )


def evaluate_matrioshka_mandelbulb_affordance(
    task_complexity: float = 1.0,
    dt: float = 1.0,
    demand_dim: int = 32
) -> Dict[str, Any]:
    """Helper function to position Matrioshka Mandelbulb compute affordances."""
    return HardwareMonitor.get_instance().evaluate_matrioshka_mandelbulb_affordance(
        task_complexity=task_complexity,
        dt=dt,
        demand_dim=demand_dim
    )

def can_afford_matrioshka_partitioning(task_complexity: float = 1.0, dt: float = 1.0) -> bool:
    """Predicate indicating whether the system can afford Matrioshka Mandelbulb partitioning."""
    res = evaluate_matrioshka_mandelbulb_affordance(task_complexity=task_complexity, dt=dt)
    return bool(res.get("can_afford_partitioning", False))

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

def get_compute_affordance_telemetry() -> Dict[str, Any]:
    """Helper function to retrieve compute affordance telemetry including gauges and typed statuses."""
    hm = HardwareMonitor.get_instance()
    gauges = hm.get_affordance_gauges()
    affordances = {t.value: bool(hm.get_typed_affordance(t)) for t in ComputeAffordanceType}
    return {
        "status": "ok",
        "has_compute_affordance": has_compute_affordance(),
        "gauges": gauges,
        "affordances": affordances,
        "thermal": get_thermal_affordance(),
        "headroom_dim": get_headroom_dimensionality()
    }

check_compute_affordance = has_compute_affordance
get_typed_compute_affordance = get_typed_affordance

def get_cpu_gpu_layout() -> Dict[str, Any]:
    """Helper function to retrieve CPU and GPU core layout topology."""
    return HardwareMonitor.get_instance().get_cpu_gpu_layout()

get_hardware_core_layout = get_cpu_gpu_layout

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
