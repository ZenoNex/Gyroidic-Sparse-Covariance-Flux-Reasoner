import psutil
import os
import math
import time
import sys
from enum import Enum
from typing import Dict, Any, Optional, Union, List

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
        
        # 1. CPU Layout via psutil + platform/brand profile
        try:
            import platform
            phys = psutil.cpu_count(logical=False) or 1
            logi = psutil.cpu_count(logical=True) or phys
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            cpu_brand = platform.processor() or ""
            layout["cpu"]["physical_cores"] = phys
            layout["cpu"]["logical_cores"] = logi
            layout["cpu"]["per_core_percent"] = per_cpu
            layout["cpu"]["brand"] = cpu_brand
            # Detect Alder Lake / Raptor Lake hybrid P+E topology (e.g. HP ENVY x360 15-ew0xxx 10-core i5-1235U/i7-1255U: 2P+8E, 12 logical)
            if phys >= 10 and logi > phys and logi < 2 * phys:
                layout["cpu"]["is_heterogeneous"] = True
                p_count = logi - phys
                layout["cpu"]["performance_cores"] = list(range(p_count))
                layout["cpu"]["efficiency_cores"] = list(range(p_count, phys))
            
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

    def detect_storage_and_hardware_profile(self, data_dir: str = "data/encodings") -> Dict[str, Any]:
        """
        Detects host laptop hardware & storage bus profile to tune fossil loading,
        SQLite WAL mmap budgets, and parallel I/O concurrency without thrashing HDDs
        or starving NVMe Gen3/Gen4 drives:
          - Acer Nitro 5 (GTX 1050 Ti 4GB VRAM, 4C/8T i7-6700HQ/7700HQ, M.2 NVMe Gen3 x4 or 2.5" SATA III)
          - HP ENVY x360 2-in-1 15-ew0xxx (12th-Gen Alder Lake 10C/12T 2P+8E, Intel Iris Xe, DDR4-3200, M.2 PCIe Gen4 x4 NVMe)
        """
        cached = getattr(self, "_cached_storage_profile", None)
        if cached is not None and cached.get("data_dir") == data_dir:
            return cached

        phys_cores = psutil.cpu_count(logical=False) or 4
        logi_cores = psutil.cpu_count(logical=True) or phys_cores
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

        has_cuda = False
        gpu_name = "CPU/Integrated"
        vram_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                has_cuda = True
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                vram_gb = round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            pass

        # Classify storage medium (NVMe vs SATA SSD vs HDD) via Windows PowerShell CIM/WMI or latency probe
        storage_bus = "UNKNOWN"
        media_type = "SSD"
        if sys.platform == "win32":
            try:
                import subprocess
                cmd = [
                    "powershell", "-NoProfile", "-Command",
                    "Get-PhysicalDisk | Select-Object -First 2 BusType,MediaType | ConvertTo-Json -Compress"
                ]
                out = subprocess.check_output(cmd, timeout=2.0, stderr=subprocess.DEVNULL, text=True).strip()
                if out:
                    import json as _json
                    disks = _json.loads(out)
                    if isinstance(disks, dict):
                        disks = [disks]
                    buses = [str(d.get("BusType", "")).upper() for d in disks]
                    medias = [str(d.get("MediaType", "")).upper() for d in disks]
                    if any("NVME" in b or b == "17" for b in buses):
                        storage_bus = "NVME"
                        media_type = "SSD"
                    elif any("SATA" in b or b == "11" for b in buses):
                        storage_bus = "SATA"
                        media_type = "HDD" if any("HDD" in m or m == "3" for m in medias) else "SSD"
            except Exception:
                pass

        # Determine named laptop profile
        is_alder_lake_envy = (phys_cores >= 10 and logi_cores > phys_cores and not has_cuda)
        is_nitro5_pascal = (has_cuda and ("1050" in gpu_name or vram_gb <= 4.5))
        if is_alder_lake_envy:
            profile_name = "HP_ENVY_X360_15_EW0_ALDER_LAKE_NVME_GEN4"
            io_workers = min(8, max(4, phys_cores - 2))
            sqlite_mmap_bytes = 268435456  # 256 MB mmap for PCIe Gen4 NVMe
            default_fossil_budget = 2500
        elif is_nitro5_pascal:
            profile_name = "ACER_NITRO_5_GTX1050TI_NVME_SATA"
            io_workers = 1 if media_type == "HDD" else min(4, max(2, phys_cores // 2))
            sqlite_mmap_bytes = 0 if media_type == "HDD" else 134217728  # 128 MB on NVMe/SATA SSD, 0 on HDD
            default_fossil_budget = 1500
        else:
            profile_name = f"GENERIC_{storage_bus}_{media_type}"
            io_workers = 1 if media_type == "HDD" else min(6, max(2, phys_cores // 2))
            sqlite_mmap_bytes = 0 if media_type == "HDD" else 134217728
            default_fossil_budget = 1500

        profile = {
            "data_dir": data_dir,
            "profile_name": profile_name,
            "physical_cores": phys_cores,
            "logical_cores": logi_cores,
            "ram_gb": ram_gb,
            "has_cuda": has_cuda,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "storage_bus": storage_bus,
            "media_type": media_type,
            "io_workers": io_workers,
            "sqlite_mmap_bytes": sqlite_mmap_bytes,
            "default_fossil_budget": default_fossil_budget,
        }
        self._cached_storage_profile = profile
        return profile

    def budget_cpu_threads(self, max_threads: Optional[int] = None) -> int:
        """
        Budgets PyTorch and MKL thread pools to prevent pegging 100% of CPU physical cores
        and freezing the host system during interactive generation loops.
        Scales up to 4 threads on 10+ core Alder Lake (HP ENVY x360 15-ew0xxx) while staying
        at <=2 threads on 4-core Acer Nitro 5 hosts.
        """
        import torch
        if max_threads is None:
            phys = psutil.cpu_count(logical=False) or 4
            cap = 4 if phys >= 10 else 2
            max_threads = max(1, min(cap, phys // 2))
        try:
            torch.set_num_threads(max_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        return max_threads

    def audit_windows_task_scheduling(self) -> Dict[str, Any]:
        """
        Audits Windows task scheduling parameters and process/thread priority states via psutil.
        Monitors thread contention, priority class, CPU core affinity, I/O rates, and handle pressure
        to ensure interactive UI serving threads (diegetic terminal, 3D graph) are not starved
        by compute-intensive background loops.
        """
        telemetry = {
            "platform": sys.platform,
            "pid": os.getpid(),
            "priority_class_raw": 32,
            "priority_class_name": "NORMAL",
            "cpu_affinity": [],
            "thread_count": 1,
            "handle_count": 0,
            "ionice": None,
            "ctx_switches": {"voluntary": 0, "involuntary": 0},
            "io_counters": {},
            "scheduler_admissibility": 1.0,
            "status": "nominal"
        }
        try:
            p = psutil.Process()
            # 1. Process priority class
            nice_val = p.nice()
            nice_int = int(getattr(nice_val, 'value', nice_val))
            telemetry["priority_class_raw"] = nice_int
            win_prio_map = {
                64: "IDLE",
                16384: "BELOW_NORMAL",
                32: "NORMAL",
                32768: "ABOVE_NORMAL",
                128: "HIGH",
                256: "REALTIME"
            }
            telemetry["priority_class_name"] = win_prio_map.get(nice_int, str(nice_val))

            # 2. CPU affinity
            try:
                telemetry["cpu_affinity"] = p.cpu_affinity()
            except Exception:
                telemetry["cpu_affinity"] = list(range(psutil.cpu_count(logical=True) or 1))

            # 3. Thread count and context switches
            telemetry["thread_count"] = p.num_threads()
            try:
                ctx = p.num_ctx_switches()
                telemetry["ctx_switches"] = {
                    "voluntary": ctx.voluntary,
                    "involuntary": ctx.involuntary
                }
            except Exception:
                pass

            # 4. Windows handle count
            if hasattr(p, "num_handles"):
                try:
                    telemetry["handle_count"] = p.num_handles()
                except Exception:
                    pass

            # 5. I/O priority & counters
            if hasattr(p, "ionice"):
                try:
                    io_val = p.ionice()
                    telemetry["ionice"] = int(getattr(io_val, 'value', io_val))
                except Exception:
                    pass
            try:
                io = p.io_counters()
                telemetry["io_counters"] = {
                    "read_count": io.read_count,
                    "write_count": io.write_count,
                    "read_bytes": io.read_bytes,
                    "write_bytes": io.write_bytes
                }
            except Exception:
                pass

            # 6. Contention heuristic & scheduler admissibility
            phys_cores = psutil.cpu_count(logical=False) or 1
            thread_ratio = telemetry["thread_count"] / max(1, phys_cores)
            if nice_int in (64, 16384):  # IDLE or BELOW_NORMAL
                telemetry["scheduler_admissibility"] = 0.5
                telemetry["status"] = "throttled_background"
            elif thread_ratio > 8.0:
                telemetry["scheduler_admissibility"] = max(0.4, 1.0 - (thread_ratio - 8.0) * 0.05)
                telemetry["status"] = "high_thread_contention"
            else:
                telemetry["scheduler_admissibility"] = 1.0
                telemetry["status"] = "nominal"

        except Exception as e:
            telemetry["error"] = str(e)
            telemetry["scheduler_admissibility"] = 0.8

        return telemetry

    def govern_task_priority(
        self,
        interactive: bool = True,
        target_priority: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Governs process and thread priority via psutil.
        When interactive=True, ensures process priority is at least NORMAL_PRIORITY_CLASS.
        When interactive=False, lowers to BELOW_NORMAL_PRIORITY_CLASS for background batching.
        """
        report = {"success": False, "previous_priority": None, "new_priority": None}
        try:
            p = psutil.Process()
            prev = p.nice()
            report["previous_priority"] = prev

            target_class = getattr(psutil, "NORMAL_PRIORITY_CLASS", 32)
            if target_priority:
                t_upper = target_priority.upper()
                prio_names = {
                    "IDLE": getattr(psutil, "IDLE_PRIORITY_CLASS", 64),
                    "BELOW_NORMAL": getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 16384),
                    "NORMAL": getattr(psutil, "NORMAL_PRIORITY_CLASS", 32),
                    "ABOVE_NORMAL": getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", 32768),
                    "HIGH": getattr(psutil, "HIGH_PRIORITY_CLASS", 128),
                }
                target_class = prio_names.get(t_upper, target_class)
            elif not interactive:
                target_class = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 16384)
            else:
                target_class = getattr(psutil, "NORMAL_PRIORITY_CLASS", 32)

            p.nice(target_class)
            report["new_priority"] = p.nice()
            report["success"] = True
        except Exception as e:
            report["error"] = str(e)
            report["new_priority"] = report["previous_priority"]
        return report

    def get_scheduler_telemetry(self) -> Dict[str, Any]:
        """Returns consolidated Windows task scheduling and process contention telemetry."""
        audit = self.audit_windows_task_scheduling()
        return {
            "status": "ok",
            "task_scheduling": audit,
            "thermal_guard_active": not self.thermal_runaway,
            "recommended_batch_concurrency": max(1, min(4, (psutil.cpu_count(logical=False) or 2) // 2))
        }

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
          2. All 8 ComputeAffordanceType dimensions (Silicon, Thermal, GPU Sovereign,
             Topological, P2P Mesh, Diegetic, Algorithmic, Compute Batch)
          3. Physical RAM and VRAM memory headroom feelers (available GB, swap pressure)
          4. Thermodynamic velocity feeler (dP_waste/dt thermal gradient)
          5. Community P2P Mesh (BonfireNomadicRing / FreenetGhostCaller) federation headroom
          6. Dynamic execution recommendations for covariance dimension, 3D render distance,
             and procedural node budgets.
        """
        import time
        now = time.time()
        if hasattr(self, '_feelers_cache') and self._feelers_cache is not None:
            if now - getattr(self, '_feelers_cache_time', 0.0) < 1.0:
                return self._feelers_cache
        t0_ns = time.perf_counter_ns()
        # Non-allocating L1/DRAM cache-line stride touch to sense t_RFC refresh stall
        _accum = 0
        for i in range(64):
            _accum ^= (i * 2654435761) & 0xFFFFFFFF
        t_rfc_feeler_ns = max(1, time.perf_counter_ns() - t0_ns)
        # Typical unstalled loop is < 50,000 ns; DRAM t_RFC / OS scheduling stall spikes higher
        t_rfc_stall_detected = bool(t_rfc_feeler_ns > 150_000)

        # 8-Regime Typed Compute Affordance Gauges
        silicon_aff = self.get_typed_affordance(ComputeAffordanceType.SILICON)
        thermal_aff = self.get_typed_affordance(ComputeAffordanceType.THERMAL)
        gpu_aff = self.get_typed_affordance(ComputeAffordanceType.GPU_SOVEREIGN)
        topo_aff = self.get_typed_affordance(ComputeAffordanceType.TOPOLOGICAL)
        mesh_aff = self.get_typed_affordance(ComputeAffordanceType.P2P_MESH)
        diegetic_aff = self.get_typed_affordance(ComputeAffordanceType.DIEGETIC)
        algorithmic_aff = self.get_typed_affordance(ComputeAffordanceType.ALGORITHMIC)
        batch_aff = self.get_typed_affordance(ComputeAffordanceType.COMPUTE_BATCH)

        # Physical RAM & Memory Pressure Feelers
        try:
            mem = psutil.virtual_memory()
            ram_available_gb = float(mem.available / (1024 ** 3))
            ram_percent = float(mem.percent)
            swap = psutil.swap_memory()
            swap_percent = float(swap.percent)
        except Exception:
            ram_available_gb = 4.0
            ram_percent = 50.0
            swap_percent = 10.0

        memory_pressure = bool(ram_percent > 88.0 or swap_percent > 75.0 or ram_available_gb < 1.0)

        # PyTorch / GPU Memory Headroom Feelers
        vram_allocated_mb = 0.0
        vram_free_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_allocated_mb = float(torch.cuda.memory_allocated() / (1024 ** 2))
                vram_free_mb = float(torch.cuda.mem_get_info()[0] / (1024 ** 2))
        except Exception:
            pass

        # Thermodynamic Gradient Feeler (dP_waste/dt)
        last_t = getattr(self, "_last_feeler_time", 0.0)
        last_p = getattr(self, "_last_p_waste", self.p_waste)
        dt = max(1e-3, now - last_t) if last_t > 0 else 1.0
        thermal_velocity = max(0.0, (self.p_waste - last_p) / dt) if last_t > 0 else 0.0
        self._last_feeler_time = now
        self._last_p_waste = self.p_waste
        thermal_spike_detected = bool(thermal_velocity > 40.0 or self.thermal_runaway or thermal_aff.gauge < 0.25)

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
            and (mesh_aff.gauge > silicon_aff.gauge * 1.15 or thermal_aff.gauge < 0.35 or memory_pressure)
        )

        # Contextual Adaptive Recommendations for Downstream / Upstream Subsystems
        if algorithmic_aff.gauge < 0.40 or memory_pressure:
            rec_cov_dim = 64
        elif algorithmic_aff.gauge < 0.70:
            rec_cov_dim = 128
        else:
            rec_cov_dim = 256

        if t_rfc_stall_detected or thermal_spike_detected or memory_pressure:
            rec_render_dist = 450.0
            rec_max_nodes = 350
            rec_mandelbulb_power = 4
        elif gpu_aff.gauge > 0.75 and thermal_aff.gauge > 0.65:
            rec_render_dist = 1200.0
            rec_max_nodes = 2500
            rec_mandelbulb_power = 8
        else:
            rec_render_dist = 800.0
            rec_max_nodes = 1200
            rec_mandelbulb_power = 6

        feelers = {
            "t_rfc_feeler_ns": int(t_rfc_feeler_ns),
            "t_rfc_stall_detected": t_rfc_stall_detected,
            "local_silicon_gauge": float(silicon_aff.gauge),
            "local_thermal_gauge": float(thermal_aff.gauge),
            "local_gpu_sovereign_gauge": float(gpu_aff.gauge),
            "topological_gauge": float(topo_aff.gauge),
            "community_p2p_mesh_gauge": float(mesh_aff.gauge),
            "diegetic_gauge": float(diegetic_aff.gauge),
            "algorithmic_gauge": float(algorithmic_aff.gauge),
            "compute_batch_gauge": float(batch_aff.gauge),
            "ram_available_gb": round(ram_available_gb, 2),
            "ram_percent": round(ram_percent, 1),
            "swap_percent": round(swap_percent, 1),
            "memory_pressure_detected": memory_pressure,
            "vram_allocated_mb": round(vram_allocated_mb, 1),
            "vram_free_mb": round(vram_free_mb, 1),
            "thermal_velocity_w_per_s": round(thermal_velocity, 2),
            "thermal_spike_detected": thermal_spike_detected,
            "freenet_online": bool(getattr(self, "freenet_online", False)),
            "active_mesh_peers": int(getattr(self, "active_mesh_peers", 0)),
            "ghost_latency_ms": float(getattr(self, "ghost_latency_ms", 0.0)),
            "kelly_consensus_headroom": float(getattr(self, "kelly_consensus_headroom", 0.5)),
            "local_to_community_gauge": float(local_to_community_gauge),
            "community_to_local_gauge": float(community_to_local_gauge),
            "community_offload_recommended": community_offload_recommended,
            "recommended_covariance_dim": rec_cov_dim,
            "recommended_procedural_render_dist": rec_render_dist,
            "recommended_procedural_max_nodes": rec_max_nodes,
            "recommended_mandelbulb_power": rec_mandelbulb_power,
        }
        self._feelers_cache = feelers
        self._feelers_cache_time = now
        return feelers

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

def audit_windows_task_scheduling() -> Dict[str, Any]:
    """Helper function to audit Windows task scheduling and thread contention telemetry."""
    return HardwareMonitor.get_instance().audit_windows_task_scheduling()

def govern_task_priority(interactive: bool = True, target_priority: Optional[str] = None) -> Dict[str, Any]:
    """Helper function to govern Windows process and thread priority."""
    return HardwareMonitor.get_instance().govern_task_priority(interactive=interactive, target_priority=target_priority)

def get_scheduler_telemetry() -> Dict[str, Any]:
    """Helper function to retrieve consolidated Windows task scheduling and process contention telemetry."""
    return HardwareMonitor.get_instance().get_scheduler_telemetry()

def budget_cpu_threads(max_threads: Optional[int] = None) -> int:
    """Helper function to budget CPU threads across PyTorch and MKL."""
    return HardwareMonitor.get_instance().budget_cpu_threads(max_threads=max_threads)


def get_affordance_budget(
    affordance_type: Union[str, ComputeAffordanceType],
    base_steps: int,
    min_steps: int = 1,
) -> tuple:
    """
    Canonical typed step/iteration budget scaler.
    Queries the exact ComputeAffordanceType domain without scalarization collapse
    and scales `base_steps` into `[min_steps, base_steps]` proportional to the
    domain gauge (falling back to `min_steps` if inadmissible or thermal runaway).

    Returns:
        (scaled_steps: int, affordance: TypedComputeAffordance)
    """
    hm = HardwareMonitor.get_instance()
    atype = ComputeAffordanceType.from_str(affordance_type)
    aff = hm.get_typed_affordance(atype)
    hi = max(1, int(base_steps))
    lo = max(1, min(int(min_steps), hi))
    if hm.thermal_runaway or not aff.is_admissible:
        return lo, aff
    scaled = int(round(lo + (hi - lo) * aff.gauge))
    return max(lo, min(hi, scaled)), aff


def should_offload_gpu_sovereign(dim: int = 8, min_dim: int = 8) -> tuple:
    """
    Checks whether PyOpenCL SiliconSovereigntyEngine GPU offload is admissible
    for a tensor operation of size `dim`, respecting both GPU_SOVEREIGN and
    THERMAL affordance domains.

    Returns:
        (admissible: bool, gpu_affordance: TypedComputeAffordance)
    """
    hm = HardwareMonitor.get_instance()
    gpu_aff = hm.get_typed_affordance(ComputeAffordanceType.GPU_SOVEREIGN)
    if dim < min_dim or hm.thermal_runaway:
        return False, gpu_aff
    thermal_aff = hm.get_typed_affordance(ComputeAffordanceType.THERMAL)
    return bool(gpu_aff.is_admissible and thermal_aff.is_admissible), gpu_aff


def get_tri_accelerator_dense_limit(cpu_limit: int = 64, gpu_limit: int = 256) -> int:
    """
    Returns the dynamic dense-linear-algebra dimension ceiling respecting the
    Tri-Accelerator Architecture (PyBevy Rust/wgpu ECS + TorchVision/CUDA +
    PyOpenCL TailSlayer Queue A/B).

    Instead of pessimistically capping at CPU-only limits (e.g. 64), raises the
    ceiling up to `gpu_limit` (default 256) whenever GPU_SOVEREIGN or CUDA is
    active and THERMAL/ALGORITHMIC affordances are admissible; degrades smoothly
    toward `cpu_limit` only on pure-CPU hosts or thermal/algorithmic pressure.
    """
    hm = HardwareMonitor.get_instance()
    if hm.thermal_runaway:
        return int(cpu_limit)
    algo_aff = hm.get_typed_affordance(ComputeAffordanceType.ALGORITHMIC)
    gpu_aff = hm.get_typed_affordance(ComputeAffordanceType.GPU_SOVEREIGN)
    thermal_aff = hm.get_typed_affordance(ComputeAffordanceType.THERMAL)
    has_cuda = False
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if (gpu_aff.is_admissible or has_cuda) and thermal_aff.is_admissible and algo_aff.is_admissible:
        scaled = int(round(cpu_limit + (gpu_limit - cpu_limit) * max(gpu_aff.gauge, algo_aff.gauge)))
        return max(int(cpu_limit), min(int(gpu_limit), scaled))
    return int(cpu_limit)


def compute_energy_based_affordance_landscape(
    dwell_times: Optional[Dict[str, float]] = None,
    frequencies: Optional[Dict[str, float]] = None,
    temperature: float = 0.25,
    reward_field: Optional[Dict[str, float]] = None,
    beta_reward: float = 1.0,
    tau_0: float = 1.0,
    affordance_gradients: Optional[Dict[str, float]] = None,
    energy_monitor: Optional[Any] = None,
    codes_framework: Optional[Any] = None,
    valence_functional: Optional[Any] = None,
    soliton_healer: Optional[Any] = None,
    moment_transport: Optional[Any] = None,
    state: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Unified Energy-Based Model (EBM) of Affordance & Codebase EBM Synthesis:
    Synthesizes:
      1. J.J. Gibson Ecological Affordances (mode parameter a in A governing regime dynamics)
         & the 7 Semantic Affordance Gradients (executability, formal symbols, runtime expandability,
         referential closure, conversational pressure, API extraction, constraint forcing),
      2. Takahashi & El Hady Statistical Mechanics of Spontaneous Behavior (docs/2611.02220v1.pdf):
         - Temperature T := sigma^2,
         - Exit-barrier Boltzmann dwell energy E(a) := -J_bar^a = -T * ln(m(a) / tau_0),
         - Mean dwell m(a) = tau_0 * exp(-E(a) / T),
         - Reward-tilted Gibbs occupation p_beta(a) proportional to m(a) * lambda(a) * exp(beta * Psi(a)),
         - Per-excursion free energy F = -T * ln(Z_0 / tau_0) where Z_0 = sum_a m(a) * lambda(a),
      3. Friston Active Inference / Markov Blanket Variational Free Energy:
         F_variational = <E(a)>_{p_beta} - T * S_Gibbs(p_beta || lambda),
      4. Canonical Codebase Energy-Based Learning Systems:
         - StructuralEnergyMonitor (LeCun et al. 2006 EBM contrastive loss functionals & Condition 3 margin),
         - CODESConstraintFramework (Georgii-Nguyen-Zessin partition-free local insertion energy, docs/1701.08105v2.pdf),
         - ValenceFunctional (Negempirical Hunger V & metaphysical entropy bands H_d + H_s + H_m),
         - EnergyBasedSolitonHealer (quadratic/linear soliton energy surface & healing efficiency),
         - ConjugateMomentTransport (ICNN Monge-Ampere EBM log-density without MCMC chains).
    """
    hm = HardwareMonitor.get_instance()
    gauges = hm.get_affordance_gauges()

    # Merge semantic affordance gradients into reward field if supplied
    if affordance_gradients and isinstance(affordance_gradients, dict):
        numeric_grads = [
            float(v) for v in affordance_gradients.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        sem_mean = sum(numeric_grads) / max(1, len(numeric_grads)) if numeric_grads else 0.0
    else:
        sem_mean = 0.0

    # Default dwell times m(a) derived from live typed affordance gauges if not supplied
    if dwell_times is None:
        dwell_times = {
            k: max(0.05, float(tau_0) * math.exp(2.0 * float(g) + 0.5 * sem_mean))
            for k, g in gauges.items()
        }
    modes = sorted(dwell_times.keys())
    n_modes = max(1, len(modes))

    if frequencies is None:
        frequencies = {k: 1.0 / float(n_modes) for k in modes}
    else:
        tot_f = max(1e-8, sum(max(0.0, float(frequencies.get(k, 0.0))) for k in modes))
        frequencies = {k: max(1e-6, float(frequencies.get(k, 0.0))) / tot_f for k in modes}

    if reward_field is None:
        reward_field = {k: float(gauges.get(k, 0.5)) + 0.25 * sem_mean for k in modes}

    t_safe = max(1e-4, float(temperature))
    tau_safe = max(1e-4, float(tau_0))

    energies: Dict[str, float] = {}
    unrewarded_weights: Dict[str, float] = {}
    tilted_weights: Dict[str, float] = {}

    for k in modes:
        m_a = max(1e-4, float(dwell_times[k]))
        lam_a = float(frequencies[k])
        psi_a = float(reward_field.get(k, 0.0))
        # E(a) = -J_bar^a = -T * ln(m(a) / tau_0)
        e_a = -t_safe * math.log(m_a / tau_safe)
        energies[k] = float(e_a)
        unrewarded_weights[k] = m_a * lam_a
        tilted_weights[k] = m_a * lam_a * math.exp(float(beta_reward) * psi_a)

    z_0 = max(1e-8, sum(unrewarded_weights.values()))
    z_beta = max(1e-8, sum(tilted_weights.values()))

    p_0 = {k: unrewarded_weights[k] / z_0 for k in modes}
    p_beta = {k: tilted_weights[k] / z_beta for k in modes}

    # Per-excursion free energy F = -T * ln(Z_0 / tau_0)
    free_energy_f = -t_safe * math.log(z_0 / tau_safe)

    # Mean energy U = <E>_{p_0} and Gibbs relative entropy S = -D_KL(p_0 || lambda)
    mean_energy_u = sum(p_0[k] * energies[k] for k in modes)
    gibbs_entropy_s = -sum(
        p_0[k] * math.log(max(1e-12, p_0[k] / max(1e-12, frequencies[k])))
        for k in modes
    )
    # First-law check: F = U - T * S
    variational_free_energy = mean_energy_u - t_safe * gibbs_entropy_s

    dominant_mode = min(modes, key=lambda k: energies[k])

    # Couple with existing Energy-Based Learning systems when provided
    coupled_ebm: Dict[str, Any] = {}
    try:
        import torch
        s_tensor = None
        if state is not None and isinstance(state, torch.Tensor):
            s_tensor = state.detach().float()
            if s_tensor.dim() == 1:
                s_tensor = s_tensor.unsqueeze(0)

        if energy_monitor is not None and hasattr(energy_monitor, "compute_ebm_loss_functionals"):
            e_vals = sorted(energies.values())
            e_c = torch.tensor([e_vals[0]], dtype=torch.float32)
            e_i = torch.tensor([e_vals[1] if len(e_vals) > 1 else e_vals[0] + 0.25], dtype=torch.float32)
            all_e = torch.tensor([e_vals], dtype=torch.float32)
            ebm_func = energy_monitor.compute_ebm_loss_functionals(e_c, e_i, all_energies=all_e)
            coupled_ebm["structural_energy_monitor"] = {
                k: v for k, v in ebm_func.items() if k != "tensors"
            }

        if codes_framework is not None and s_tensor is not None:
            c_dim = getattr(codes_framework, "state_dim", s_tensor.shape[-1])
            s_codes = torch.nn.functional.pad(s_tensor.reshape(1, -1), (0, max(0, c_dim - s_tensor.numel())))[:, :c_dim]
            tot_e = float(codes_framework.compute_total_energy(s_codes).mean().item())
            gnz = codes_framework.compute_gnz_local_insertion_energy(s_codes)
            coupled_ebm["codes_constraint_framework"] = {
                "total_constraint_energy": tot_e,
                "gnz_local_insertion_energy": float(gnz.get("mean_insertion_energy_h", 0.0)),
                "estimated_beta_gnz": float(gnz.get("estimated_beta_gnz", 1.0)),
            }

        if valence_functional is not None:
            p_t = torch.tensor(abs(mean_energy_u), dtype=torch.float32)
            hunger = float(valence_functional(p_t).mean().item())
            coupled_ebm["valence_functional"] = {
                "negempirical_hunger_V": hunger,
                "satisfaction_baseline": float(valence_functional.satisfaction.item()),
            }

        if soliton_healer is not None and s_tensor is not None:
            h_dim = getattr(soliton_healer, "state_dim", s_tensor.shape[-1])
            s_heal = torch.nn.functional.pad(s_tensor.reshape(1, -1), (0, max(0, h_dim - s_tensor.numel())))[:, :h_dim]
            sol_e = float(soliton_healer.compute_energy(s_heal).mean().item())
            coupled_ebm["soliton_healer"] = {
                "soliton_energy": sol_e,
                "stability_metrics": soliton_healer.get_stability_metrics(),
            }

        if moment_transport is not None and s_tensor is not None and hasattr(moment_transport, "compute_monge_ampere_ebm_density_loss"):
            m_dim = getattr(moment_transport, "dim", s_tensor.shape[-1])
            s_mt = torch.nn.functional.pad(s_tensor.reshape(1, -1), (0, max(0, m_dim - s_tensor.numel())))[:, :m_dim]
            coupled_ebm["conjugate_moment_transport"] = moment_transport.compute_monge_ampere_ebm_density_loss(s_mt)
    except Exception as e:
        coupled_ebm["coupling_warning"] = str(e)

    return {
        "temperature_T": t_safe,
        "partition_mean_cycle_time_Z0": float(z_0),
        "per_excursion_free_energy_F": float(free_energy_f),
        "variational_free_energy": float(variational_free_energy),
        "first_law_residual": float(abs(free_energy_f - variational_free_energy)),
        "mean_energy_U": float(mean_energy_u),
        "gibbs_entropy_S": float(gibbs_entropy_s),
        "energies_E": energies,
        "occupation_p0": p_0,
        "reward_tilted_occupation_p_beta": p_beta,
        "dominant_ground_state_affordance": dominant_mode,
        "semantic_affordance_gradients": affordance_gradients or {},
        "coupled_ebm_systems": coupled_ebm,
    }




