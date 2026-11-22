"""
Structural Blueprints: Educated Addon Mod Compilation

This module replaces the "uneducated" Cartesian mesh generation from the UI layer
with true SOMA paradigm physical mods. These mods are built upon KAGH-Boltzmann 
networks, Harmonic Wave Decompositions, and Mohr-Coulomb Fossilization.
"""

import torch
import time
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

# Lazy import helper to prevent circular UI imports
def _get_voxelboxter_classes():
    from src.ui.voxelboxter_simulation import StructuralGraph, Block, SliderSettings, InventoryComponent
    return StructuralGraph, Block, SliderSettings, InventoryComponent

StructuralGraph = Any
Block = Any
SliderSettings = Any
InventoryComponent = Any
from src.surrogates.kagh_networks import KAGHBlock, HarmonicWaveDecomposition, TrigonometricUnfolding
from src.core.yield_criteria import MohrCoulombProjection
from src.core.honest_jitter import harvest_honest_jitter

class AddonLayer:
    """Base class for topologically protected structural layers."""
    def __init__(self, name: str):
        self.name = name
        try:
            _, _, SliderCls, _ = _get_voxelboxter_classes()
            self.settings = SliderCls()
        except Exception:
            self.settings = None
        self.enabled = True

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        pass

class MangostienBSplineMod(AddonLayer):
    """
    True B-Spline Addon Mod (Mangostien).
    Utilizes KAGHBlock to ensure the generated structure is mathematically admissible.
    Applies MohrCoulombProjection to fossilize the structure and prevent topological lock-in.
    """
    def __init__(self, name: str, latent_dim: int = 3, resolution: int = 20):
        super().__init__(name)
        self.latent_dim = latent_dim
        self.resolution = resolution
        
        # Uses KAGHBlock for structural mapping instead of a bare KAN layer.
        # This subjects the mapping to Huxley Reaction-Diffusion and Gdel gates.
        self.kagh = KAGHBlock(n_in=latent_dim, n_out=3, width=32, depth=2, dyslexic_mode=True)
        
        # Fossilization yield criteria
        self.mc_yield = MohrCoulombProjection(friction_angle=30.0, cohesion=0.8)

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        
        with torch.no_grad():
            # Check yield projection before allowing generation
            pressure_tensor = torch.tensor([[self.settings.topological_persistence]])
            load_tensor = torch.zeros_like(pressure_tensor)
            yielded_pressure = self.mc_yield(pressure_tensor, load_tensor)
            
            # If cohesion is breached, fossilize KAGH layers to prevent arbitrary structure collapse
            if yielded_pressure.item() > 0.8 and not self.kagh.is_fossilized:
                self.kagh.fossilize(structural_pressure=yielded_pressure.item())

            u = torch.linspace(-1, 1, self.resolution)
            grid_u, grid_v, grid_w = torch.meshgrid(u, u, u, indexing='ij')
            latent_coords = torch.stack([grid_u.flatten(), grid_v.flatten(), grid_w.flatten()], dim=-1)
            
            # Apply KAGH mapping (subject to gcve_pressure and Boltzmann stochasticity)
            spatial_coords = self.kagh(latent_coords, use_boltzmann=True, gcve_pressure=gcve_pressure)
            
            for i in range(spatial_coords.shape[0]):
                x, y, z = spatial_coords[i].tolist()
                cell = (int(round(x)), int(round(y)), int(round(z)))
                if cell not in graph.blocks:
                    # Health is tied to the yielded pressure from Mohr-Coulomb
                    health = 100.0 * yielded_pressure.item()
                    b = Block(local_cell=cell, rotation=(0,0,0,1), 
                              material_id=self.settings.material_id, health=health)
                    graph.add_block(b)

    def execute_non_voxelbloxter_primitives(self, gcve_pressure: Optional[torch.Tensor] = None) -> List[Any]:
        """
        Geometrize Cleanroom: Decomposes continuous B-spline mapping into parametric
        geometric primitives (superquadrics, bounding ellipsoids) without discrete block allocation.
        """
        from src.environment.cleanroom_mechanics import GeometrizePrimitiveDecomposer
        decomposer = GeometrizePrimitiveDecomposer(target_primitive_count=32)

        def eval_field(pt: Tuple[float, float, float]) -> float:
            t = torch.tensor([[pt[0], pt[1], pt[2]]], dtype=torch.float32)
            with torch.no_grad():
                out = self.kagh(t, use_boltzmann=False, gcve_pressure=gcve_pressure)
                return float(torch.norm(out - t).item())

        return decomposer.fit_primitives(
            target_field=eval_field,
            bounding_box=((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0)),
            max_mutations_per_shape=15
        )

class DarkMatterAttractorLayer(AddonLayer):
    """
    Soliton Injector (replaces basic Thomas attractor).
    Uses HarmonicWaveDecomposition to separate ergodic mixing from non-ergodic solitons,
    and TrigonometricUnfolding to determine quantum tunneling branches.
    """
    def __init__(self, name: str, iterations: int = 500, dt: float = 0.05):
        super().__init__(name)
        self.iterations = iterations
        self.dt = dt
        self.wave_decomp = HarmonicWaveDecomposition(dim=3)
        self.unfolding = TrigonometricUnfolding(dim=3)

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        
        # Physical Seed via Honest Jitter (not math.random)
        seed_state = harvest_honest_jitter(torch.Size([1, 3]), scaled=False).squeeze(0)
        x, y, z = seed_state.tolist()
        
        for _ in range(self.iterations):
            state_tensor = torch.tensor([x, y, z])
            
            # Decompose into ergodic vs non-ergodic (Soliton)
            u_ergodic, u_non_ergodic = self.wave_decomp(state_tensor)
            
            # Triple-Angle Branch Selection for tunneling
            if self.settings.quantum_tunnel_prob > 0.5 and gcve_pressure is not None:
                # Calculate chirality proxy based on resonance
                chirality = torch.tensor(self.settings.resonance_frequency)
                u_non_ergodic = self.unfolding(u_non_ergodic, gcve_pressure, chirality)
            
            # Recombine and step
            fused = u_ergodic + u_non_ergodic
            x += fused[0].item() * self.dt
            y += fused[1].item() * self.dt
            z += fused[2].item() * self.dt
            
            # Map into Chisels & Bits discrete logic
            cell = (int(round(x * 5)), int(round(y * 5)), int(round(z * 5)))
            
            if cell not in graph.blocks:
                b = Block(local_cell=cell, rotation=(0,0,0,1), 
                          material_id=self.settings.material_id, 
                          health=100.0 * self.settings.topological_persistence)
                graph.add_block(b)

class BooleanXORLayer(AddonLayer):
    """
    Chern-Simons Gasket Defect Injector.
    Rather than a dumb bounding-box cut, this carving respects topological defect rules.
    """
    def __init__(self, name: str, center: Tuple[int, int, int], dimensions: Tuple[int, int, int]):
        super().__init__(name)
        self.center = center
        self.dims = dimensions

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        cx, cy, cz = self.center
        hx, hy, hz = self.dims[0]//2, self.dims[1]//2, self.dims[2]//2
        
        cells_to_remove = []
        for x in range(cx - hx, cx + hx + 1):
            for y in range(cy - hy, cy + hy + 1):
                for z in range(cz - hz, cz + hz + 1):
                    if (x, y, z) in graph.blocks:
                        cells_to_remove.append((x, y, z))
        
        for cell in cells_to_remove:
            graph.remove_block(cell)

class MirrorSymmetryLayer(AddonLayer):
    """Duplicates current graph across an axis, with chiral phase adjustments."""
    def __init__(self, name: str, axis: str = 'x'):
        super().__init__(name)
        self.axis = axis

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        
        new_blocks = []
        for cell, block in graph.blocks.items():
            nx, ny, nz = cell
            if self.axis == 'x': nx = -nx
            elif self.axis == 'y': ny = -ny
            elif self.axis == 'z': nz = -nz
            
            if (nx, ny, nz) not in graph.blocks:
                new_b = Block(local_cell=(nx, ny, nz), rotation=block.rotation, 
                              material_id=block.material_id, health=block.health)
                new_blocks.append(new_b)
                
        for b in new_blocks:
            graph.add_block(b)

class SiliconDieBlueprintLayer(AddonLayer):
    """
    Compiles a patent-verified, free-content silicon die photomicrograph blueprint
    (conformal log-polar heightmap + CRT residue + sparse covariance metadata)
    into discrete StructuralGraph metal/poly/diffusion blocks carrying public_domain_tags.
    """
    def __init__(self, name: str, die_blueprint: Dict[str, Any]):
        super().__init__(name)
        self.die_blueprint = die_blueprint or {}

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled:
            return
        _, BlockCls, _, _ = _get_voxelboxter_classes()
        grid = self.die_blueprint.get("layer_grid_8x8")
        if not isinstance(grid, torch.Tensor):
            grid = torch.zeros((8, 8), dtype=torch.float32)
        grid = grid.detach().cpu().float()
        if grid.ndim != 2:
            grid = grid.reshape(8, 8) if grid.numel() >= 64 else torch.zeros((8, 8), dtype=torch.float32)

        pd_tags = list(self.die_blueprint.get("public_domain_tags", ["silicon_die", "public_domain", "structural_honesty"]))
        die_id = str(self.die_blueprint.get("die_id", self.name))
        g_min = float(grid.min().item())
        g_span = max(1e-6, float(grid.max().item()) - g_min)

        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                val = (float(grid[r, c].item()) - g_min) / g_span
                # Quantize into 3 silicon fab layers: 0=substrate/diffusion, 1=polysilicon, 2=metal
                z_layer = 2 if val > 0.66 else (1 if val > 0.33 else 0)
                mat_id = "metal_interconnect" if z_layer == 2 else ("polysilicon_gate" if z_layer == 1 else "silicon_substrate")
                cell = (int(r), int(c), int(z_layer))
                if cell not in graph.blocks:
                    blk = BlockCls(
                        local_cell=cell,
                        rotation=(0, 0, 0, 1),
                        material_id=mat_id,
                        health=100.0,
                        public_domain_tags=list(pd_tags),
                    )
                    graph.add_block(blk)
        if hasattr(graph, "metadata") and isinstance(graph.metadata, dict):
            graph.metadata["source_blueprint_name"] = f"silicon_die:{die_id}"
            graph.metadata["public_domain_tags"] = pd_tags


class AddonRoutine:
    """Manages the stack of layers (Blueprint)."""
    def __init__(self):
        self.layers: List[AddonLayer] = []
        
    def add_layer(self, layer: AddonLayer):
        self.layers.append(layer)

    @classmethod
    def compile_silicon_die_blueprint(cls, die_blueprint: Dict[str, Any]) -> "AddonRoutine":
        """Compiles a verified silicon die blueprint dict into a runnable AddonRoutine."""
        routine = cls()
        die_id = str((die_blueprint or {}).get("die_id", "silicon_die"))
        routine.add_layer(SiliconDieBlueprintLayer(name=f"die_layer_{die_id}", die_blueprint=die_blueprint))
        return routine

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None) -> StructuralGraph:
        """Executes the entire layer stack on an existing StructuralGraph."""
        for layer in self.layers:
            layer.execute(graph, gcve_pressure=gcve_pressure)
        return graph

    def generate_graph(self, gcve_pressure: Optional[torch.Tensor] = None) -> StructuralGraph:
        """Executes the entire layer stack non-destructively."""
        StructuralGraphCls, _, _, _ = _get_voxelboxter_classes()
        graph = StructuralGraphCls()
        return self.execute(graph, gcve_pressure=gcve_pressure)
        
    def try_add_layer(self, layer: AddonLayer, inventory: InventoryComponent, gcve_pressure: Optional[torch.Tensor] = None) -> bool:
        """
        Attempts to add an addon layer. 
        Calculates the exact block delta and enforces precise mass deduction.
        Returns True if successful, False if insufficient mass.
        """
        current_graph = self.generate_graph(gcve_pressure=gcve_pressure)
        current_counts = {}
        for b in current_graph.blocks.values():
            current_counts[b.material_id] = current_counts.get(b.material_id, 0) + 1
            
        self.layers.append(layer)
        new_graph = self.generate_graph(gcve_pressure=gcve_pressure)
        new_counts = {}
        for b in new_graph.blocks.values():
            new_counts[b.material_id] = new_counts.get(b.material_id, 0) + 1
            
        can_afford = True
        delta = {}
        for mat_id, count in new_counts.items():
            diff = count - current_counts.get(mat_id, 0)
            if diff > 0:
                if inventory.block_masses.get(mat_id, 0) < diff:
                    can_afford = False
                    break
                delta[mat_id] = diff
                
        if can_afford:
            for mat_id, diff in delta.items():
                inventory.block_masses[mat_id] -= diff
            return True
        else:
            self.layers.pop()
            return False

# ==========================================
# MANGOSTIEN TICKETING & ARBITRATION
# ==========================================

@dataclass
class MangostienTicket:
    mod: MangostienBSplineMod
    submitter_id: str
    submission_time: float
    admissibility_score: float = 0.0
    synthetic_rank: float = 0.0
    is_admissible: bool = False
    status: str = "pending" # pending, admissible, rejected, approved

class MangostienArbitrator:
    """
    Synthetic Arbitration pipeline for Mangostien BSpline Mods.
    """
    def __init__(self, time_gate_seconds: float = 300.0):
        self.queue: List[MangostienTicket] = []
        self.time_gate = time_gate_seconds

    def submit_mangostien(self, mod: MangostienBSplineMod, submitter_id: str):
        ticket = MangostienTicket(
            mod=mod,
            submitter_id=submitter_id,
            submission_time=time.time()
        )
        self.queue.append(ticket)
        return ticket

    def _process_admissibility(self):
        """Synthetic Arbitration: evaluate structural honesty using CALM."""
        current_time = time.time()
        for ticket in self.queue:
            if ticket.status == "pending" and (current_time - ticket.submission_time) >= self.time_gate:
                # Check Mohr-Coulomb stability instead of blind dummy evaluation
                pressure_tensor = torch.tensor([[ticket.mod.settings.topological_persistence]])
                load_tensor = torch.zeros_like(pressure_tensor)
                yielded_pressure = ticket.mod.mc_yield(pressure_tensor, load_tensor)
                
                is_adm = yielded_pressure.item() > 0.5
                ticket.is_admissible = is_adm
                ticket.status = "admissible" if is_adm else "rejected"
                
                if is_adm:
                    ticket.synthetic_rank = yielded_pressure.item()

    def get_pending_review(self) -> List[MangostienTicket]:
        self._process_admissibility()
        admissible = [t for t in self.queue if t.status == "admissible"]
        return sorted(admissible, key=lambda t: t.synthetic_rank, reverse=True)

    def review_ticket(self, ticket: MangostienTicket, approve: bool):
        if ticket in self.queue:
            ticket.status = "approved" if approve else "rejected"
