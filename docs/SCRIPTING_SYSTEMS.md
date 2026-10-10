# Gyroidic Scripting Systems

This document explains the three distinct tiers of scripting available within the Gyroidic Sparse Covariance Flux Reasoner ecosystem. Rather than a monolithic scripting language, the system partitions control into **Semantic**, **Structural**, and **Physical** tiers, enforcing the philosophy of "Patch Sovereignty" across its biomimetic components.

---

## Tier 1: Semantic Scripting (Diegetic Terminal)

The **Diegetic Terminal** (`http://localhost:8000`) is the highest-level interface, driven by `src/ui/diegetic_terminal.html` and `src/ui/diegetic_backend.py`. It provides control over the semantic meaning and cognitive "focus" of the Reasoner.

### Capabilities
- **Manifold Injection**: Users can inject raw semantic concepts, text, or multimodal dyad associations (images, audio).
- **Control Flags**: Adjust the Reasoner's regime (GOO vs. PRICKLES) and dyad commutativity.
- **Cognitive Routing**: By associating concepts, you script the "thought process" of the Reasoner as it navigates the embedding manifold.

*For full details on semantic routing, see [INTERFACE_LAYER.md](INTERFACE_LAYER.md).*

---

## Tier 2: Structural Scripting (Voxelboxter CLI & REPL)

**Voxelboxter** (`src/ui/voxelboxter_client.py`) acts as the tangible, PyBevy-based voxel client where abstract math turns into structural physics. It features a chat-based command line interface that provides Structural Scripting.

### Commands
- `/addon bspline [latent_dim]`: Injects a Compiled B-Spline mathematical mod layer into the environment.
- `/addon node`: Launches the **Physical Node Editor** (see Tier 3).
- `/create`: Switches the engine to CREATION mode (Admin editing unlocked).
- `/play`: Switches the engine to PLAY mode (Survival logic and physics enabled).

### Sovereign Execution (`/eval` and `/exec`)
Voxelboxter enforces **Patch Sovereignty**, meaning that if you have `ADMIN` or `BUILDER` roles, you possess unrestricted access to the underlying engine state. 

- `/eval [expression]`: Evaluates a raw Python expression.
- `/exec [statement]`: Executes raw Python statements.

**Context Injected**: Both commands are injected with the active PyBevy `state` (the `PatchStateResource`), `torch`, and `logging`.

*Example:*
```python
/eval len(state.graph.blocks)
/exec state.fingerprint_energy = 100.0
```

> **Note on Security**: These commands use unrestricted `eval()` and `exec()`. This is by design. The system does not sandbox the Administrator; to restrict `locals` would limit the kind of mods that could be generated, breaking the integrity of the sovereign user.

---

## Tier 3: Physical Scripting (Node Environment)

The most direct, hardware-adjacent scripting layer is the **Physical Node Editor** (`src/scripting/node_environment.py`), built using `dearpygui`.

When you run `/addon node` in Voxelboxter, the DearPyGui node editor opens. It implements a Blender-style "Plug & Play" dataflow graph architecture that directly binds to the core mathematical and physical engines without redundant wrapper abstractions.

### 1. Dataflow Graph Architecture Under the Hood
Unlike procedural scripts that execute sequentially top-to-bottom, the node tree evaluates as a pure dataflow dependency graph:
* **Evaluation (Right-to-Left Pull):** Sinks and output nodes (such as the `Voxelboxter Graph Sink` or `Material/Group Output`) request data from their leftward upstream inputs. Each node pulls required data backwards through the chain, computes its local transformation, and passes results forward.
* **Stream (Left-to-Right Push):** Computed values flow through connected wires from output sockets on the left to input sockets on the right:
  ```
  [ Input Node ] ──(Data Out)──► [ Math / Processing Node ] ──(Data Out)──► [ Output Node ]
  (Provides values)              (Calculates transformations)             (Applies to mesh/sim)
  ```

### 2. Sockets and Wires
Nodes communicate through strictly typed sockets defined by `SocketType`:
* **Blue Sockets (`VECTOR`):** 3D positional/directional vectors `[X, Y, Z]`. Used for coordinate offsets, impulse vectors, and velocity vectors.
* **Yellow Sockets (`COLOR`):** 4-element color vectors `[R, G, B, A]` in normalized floating point or OKLab space.
* **Gray Sockets (`FLOAT`):** Single scalar values (e.g., hardness, friction, mass cost, battery charge).
* **Green / Pink Sockets (`INT` / `BOOLEAN`):** Whole numbers (material IDs, iteration counts) and boolean flags.
* **Diamond vs. Circle Sockets:**
  * **Circle Socket:** Carries a uniform scalar or vector value applied across an entire object.
  * **Diamond Socket (Field):** Carries per-vertex or per-voxel spatial field calculations evaluated across the geometry.

### 3. Master Node Groups & Game Engine Attribute Baking
To avoid rebuilding repetitive node graphs across hundreds of modular component meshes, the environment provides **Master Node Groups** (`MasterNodeGroup`):
* **Packaging and Reusability:** Collapses complex mathematical subgraphs into reusable macros with exposed sliders.
* **Center of Gravity (COG) Offset Logic:**
  * Reads the mesh dimensions from a `Bounding Box` calculation.
  * Multiplies the dimension vector by `-0.5` to calculate the exact geometric midpoint vector.
  * Offsets the geometry vertices relative to the origin so that physics pivots rest precisely dead center.
* **Named Vertex Attribute Injection:**
  * Bakes custom physical parameters (`phys_hardness`, `phys_friction`, `phys_hp`, `cog_offset`) into the 3D file's vertex data layer prior to `.glb` or `.obj` export.
  * Downstream physics engines and vehicle controllers read `phys_friction` for tire traction, `phys_hardness` for damage thresholds, and `cog_offset` for rigid body mass distribution.

### 4. Direct Subsystem Integrations (Zero Redundant Wrappers)
The node environment directly interacts with canonical engine classes:

1. **Discrete Bit Carving (`PointerlessOctree` & `BooleanXORLayer`):**
   * Carves and places 1/16th scale sub-voxel bits (4096 micro-bits per macro block) via `chisel_carve_bit` and `chisel_place_bit`.
   * Employs 3D Morton Z-order curve bit-interleaving for linear memory access and recycles carved debris into scrap inventories.
2. **Crafting & Addon Assembly (`AddonRoutine` & `InventoryComponent`):**
   * Multi-stage recipe assembly validating material costs and deducting power from the energy bus.
   * `OreDictionary` provides polymorphic resource resolution (e.g., matching `oreIron`, `ingotIron`, `gemDiamond`, `chiselBitStone`).
3. **Vehicle Kinetics & Delta-v Collision Physics (`RigidBody`, `AirBreathingBattery`, `VehicleEngine`):**
   * **Real-Time Impulse Dynamics:** Computes dynamic velocity changes frame-by-frame:
     $$\Delta v = \frac{J}{m}$$
     where $J$ is the collision impulse magnitude and $m$ is the vehicle mass.
   * **$\Delta t$ Hardness Curve:** Deceleration over a prolonged $\Delta t$ results in controlled crumpling; an identical $\Delta v$ over a near-zero $\Delta t$ causes catastrophic structural failure.
   * **Tiered Damage Thresholds:**
     * Low Tier ($\Delta v < 5\text{ mph}$): Cosmetic scuffs and surface particle dust.
     * Mid Tier ($5\text{ mph} \le \Delta v \le 15\text{ mph}$): Panel denting and localized voxel detachment.
     * High Tier ($\Delta v > 15\text{ mph}$): Catastrophic structural shearing and block ejection from inventory.
   * **Tuning Augments:**
     * `reinforced_alloy_chassis`: Increases vehicle mass and doubles structural $\Delta v$ shearing thresholds.
     * `impact_energy_harvester`: Converts high-severity kinetic impact energy directly into temporary velocity and nitro boosts.
   * **Plasma Air Induction:** `AirBreathingBattery` features plasma ionization (`plasma_air_induction = True`, 2.4x boost factor) converting ambient atmospheric $\text{N}_2$ and $\text{O}_2$ into high-energy oxidizers, bypassing environmental $\text{NO}_2$ synthesis scarcity.
   * **Sleeve-Valve Combustion Timings:** `VehicleEngine` simulates sleeve-valve electrical advance timings (12.5 degrees) and port overlap efficiencies (1.35x) to optimize power draw and thrust curves.
4. **Vitality & Life/Hunger Governor (`ValenceFunctional` & `EgoDeathThresholdMonitor`):**
   * Metabolic burn and hunger depletion scale dynamically across `DifficultyMode` (`PEACEFUL`, `SURVIVAL`, `HARSH`, `ENTROPIC`).
   * Evaluates dissonance and manifold hunger via `ValenceFunctional` and monitors non-ergodic abstraction rates ($R_a$) to safeguard against ego-death collapse.
5. **Procedural Character Rigs & Enemy Subtypes (`AdaptiveSkeletonHarness`):**
   * Generates rigged procedural skeletal structures across `AmbulatoryClass` kinematics (bipedal, hexapod, tracked, serpentine).
   * Injects `EnemySubtype` deformations (e.g., `ABSTRACTED_GLITCH` phase noise, `DEMIURGIC_TITAN` localized $dt$ dilation, `MANNEQUIN_INFILTRATOR` low-degree facet scripts).
6. **Thermodynamic Admin Shop (`CarnotMobiusLedger` & `LeontiefGovernor`):**
   * Admin and visitor trading bounded by thermodynamic waste heat dissipation ($P_{\text{waste}}$) and Carnot efficiency limits ($\eta_{\text{stack}}$).
   * Enforces Leontief input-output balance criteria with Kelly-hedged margins.

### 5. D-Wave Collective Computation Pool
To prevent UI blocking during heavy mathematical evaluations and DSP processing, the Physical Scripting tier uses an **asynchronous evaluation worker thread**:
* Evaluates node graphs in a dedicated side-chain thread while monitoring hardware headroom.
* Tracks slider states (Latent Dimension, Resolution, Mass Cost, Topological Persistence, Quantum Tunneling Probability).
* When a functional node links to the `Voxelboxter Graph Sink`, the worker dynamically compiles the layer (e.g., `MangostienBSplineMod`, `DarkMatterAttractorLayer`), signs it with honest silicon jitter, validates Chern-Simons topological invariants via `ZKAggregator`, and hot-injects it directly into the running PyBevy `PatchStateResource.routine`.

---

## Tier 4: Cleanroom Systems Suite (Industrial, Ecological & Logistical)

The Physical Scripting and Voxelboxter engines incorporate cleanroom implementations of classic sandbox, automation, and ecological mechanics, bridging them directly into the underlying topological manifold:

### 1. Spatial Reconnaissance & Waypoints (`JourneyTopoRadar`)
* **Inspiration:** JourneyMap.
* **Architecture:** Projects 3D voxel heightmaps and cavern depths into 2D topographical radar slices.
* **Death Fossils:** When an entity abstracts or dies, a permanent historical stress tensor marker (`TopoWaypoint` with $\beta = 0$) is fossilized at the death coordinate, anchoring non-ergodic memory across reloads.
* **Node Interface:** `JourneyTopoRadar (JourneyMap)` node controls radar radius, subterranean slice filtering, and entity blip output vectors.

### 2. Resource Distributions & Drop Matrices (`ResourceDistributionInspector`)
* **Inspiration:** Just Enough Resources (JER).
* **Ore Height Distributions:** Models ore spawn probability density curves across Chebyshev depth bands $z \in [-64, 320]$ for coal, iron, copper, gold, redstone, diamond, uranium, and dark matter flux.
* **Mob Drops & Dungeon Manifolds:** Configures drop chances scaled by looting levels and non-ergodic entropy ($H_{\text{mischief}}$).
* **Node Interface:** `Resource Distribution (JER)` node queries height density and drop probabilities for arbitrary material IDs.

### 3. Procedural Mob Attributes & Elite Modifiers (`MobProperties` & `InfernalAffixRegistry`)
* **Inspiration:** Mob Properties & AtomicStryker's Infernal Mobs / Multi Mine.
* **Attribute Control:** Configures max HP, movement speed, attack damage, knockback resistance, and follow range.
* **Infernal Affixes:** Elite mobs dynamically spawn with 1-4 random affixes (`1UP`, `Alchemist`, `Berserk`, `Blastoff`, `Bulwark`, `Cloaking`, `Darkness`, `Ender`, `Exhaust`, `Fiery`, `Ghastly`, `Hermetic`, `Lifesteal`, `Ninja`, `Poisonous`, `Quicksand`, `Regen`, `Rust`, `Sapper`, `Sprint`, `Storm`, `Twin`, `Webbing`, `Wither`).
* **Multi-Mine Persistent Memory:** `MultiMineMemory` tracks block fracture damage across ticks and multi-agent mining attempts, preventing tool-release damage resets.
* **Node Interface:** `Mob Properties & Infernal Affixes` node mutates entity baselines into elite adversaries.

### 4. Multi-Tier Metallurgical Dissolution (`MekanismProcessingPipeline`)
* **Inspiration:** Mekanism, Mekanism Tools, Mekanism Generators.
* **Tiered Multiplication:**
  * **Tier 1 (Smelting):** 1x yield (Ore $\to$ 1 Ingot).
  * **Tier 2 (Enrichment):** 2x yield (Ore $\to$ 2 Dust $\to$ 2 Ingots).
  * **Tier 3 (Purification):** 3x yield using Oxygen injection (Ore + $\text{O}_2 \to$ 3 Clumps $\to$ 3 Shards $\to$ 3 Dust).
  * **Tier 4 (Chemical Injection):** 4x yield using Hydrogen Chloride ($\text{HCl}$) gas.
  * **Tier 5 (Chemical Dissolution):** 5x yield using Sulfuric Acid ($\text{H}_2\text{SO}_4$) dissolution into clean slurry, followed by crystallizer precipitation.
* **Node Interface:** `Mekanism Ore Refinery (1x-5x)` node balances raw ore input against chemical reagent volumes to calculate refined metal yields.

### 5. Buoyant Airships & Propeller Flight (`AeronauticContraption`)
* **Inspiration:** Create: Aeronautics.
* **Buoyancy Physics:** Computes Archimedes buoyant force ($F_{\text{buoyant}} = (\rho_{\text{air}} - \rho_{\text{gas}}) \cdot V \cdot g$) for helium and hot-air envelopes against ambient `EnvironmentalAtmosphere`.
* **Aerodynamics:** Computes propeller thrust curves and aerodynamic drag ($F_{\text{drag}} = \frac{1}{2}\rho v^2 C_d A$) to govern rigid flying contraptions detached from the block grid.
* **Node Interface:** `Create: Aeronautics Contraption` node balances balloon volume, gas types, and RPM into net lift and thrust vectors.

### 6. Culinary Nutrition & Diet Progression (`NutritionalDiversityTracker`)
* **Inspiration:** Farmer's Delight & Spice of Life (Carrot & Onion Editions).
* **Carrot Edition (Milestone Heart Progression):** Discovering unique food items permanently increases max HP ($+2$ HP for every 5 unique foods discovered).
* **Onion Edition (Dynamic Rotational Buffs):** Evaluates Shannon diversity entropy over a rolling window of recent meals (e.g. 12 meals). High diet diversity activates temporary buffs (`SPEED_II`, `HASTE_I`), while dietary monotony triggers nutritional malaise slowness.
* **Node Interface:** `Spice of Life & Nutrition` node tracks meal history, hunger saturation, and active status effects.

### 7. Multi-Bus Conduits & 16-Channel RedNet (`CompositeVoxelConduit` & `RedNet16BundledCable`)
* **Inspiration:** EnderIO & RedNet / MineFactory Reloaded.
* **Single-Voxel Multiplexing:** Power, fluid, item, and signal conduits coexist within a single voxel coordinate space without cross-talk or block collision.
* **RedNet Bundled Subnets:** Carries 16 distinct colored redstone subnets (white, orange, magenta, light blue, yellow, lime, pink, gray, light gray, cyan, purple, blue, brown, green, red, black) with individual analog signals (0-255) through a single cable connection.
* **Node Interface:** `EnderIO & RedNet Bundled Cable` node configures channel colors and bus routing.

### 8. Automated Logistics & Directional Inserters (`LogisticsTransportNetwork`)
* **Inspiration:** Factorio & BuildCraft.
* **Dual-Lane Belts:** Models transport belts with two independent item lanes, supporting tiers from 15 items/sec up to 45 items/sec.
* **Directional Inserters:** Simulates pickup, swing timing, filter selection, and stack capacities between inventories and transport belts.
* **Node Interface:** `Factorio Logistics Network` node regulates logistics throughput across automated assembly lines.

### 9. Mendelian Animal Husbandry (`FaunaGeneticsComponent`)
* **Inspiration:** Animal Husbandry & Genetics mods.
* **Allelic Inheritance:** Chromosomal alleles for movement speed, jump height, and material yields, supporting crossover and mutation rates influenced by ambient gyroidic flux.
* **Node Interface:** `Animal Husbandry & Genetics` node monitors expressed speeds, jumps, and yields across generations.

### 10. Node Environment Execution Hooks & Telemetry Wiring (`PhysicalNodeEditor`)
The DearPyGui [node_environment.py](../../../../src/scripting/node_environment.py) executes continuous headless and interactive simulation hooks linking mechanics inside and outside of [cleanroom_mechanics.py](../../../../src/environment/cleanroom_mechanics.py):

#### Internal Cleanroom Mechanics Hooks:
* `hook_cleanroom_radar(radar_radius, show_death_fossils, center_pos)`: Ticks [JourneyTopoRadar](../../../../src/environment/cleanroom_mechanics.py#L186) subterranean scanning, detects entity blips, and tracks death fossilation waypoints.
* `hook_cleanroom_jer(material_id, y_height, looting_level)`: Evaluates [ResourceDistributionInspector](../../../../src/environment/cleanroom_mechanics.py#L249) Chebyshev orthogonal polynomial ore distribution curves and rolls physical harvest debris.
* `hook_cleanroom_multi_mine(coord, tool_power)`: Accumulates block damage in [MultiMineMemory](../../../../src/environment/cleanroom_mechanics.py#L386), fractures blocks into the [PointerlessOctree](../../../../src/ui/voxelboxter_simulation.py#L767), and yields harvested mass into [InventoryComponent](../../../../src/ui/voxelboxter_simulation.py#L52).
* `hook_cleanroom_mekanism(tier, raw_ore, chemical_reagent_mb)`: Steps [MekanismProcessingPipeline](../../../../src/environment/cleanroom_mechanics.py#L451) (1x-5x refining) while monitoring thermodynamic entropy via [CarnotMobiusLedger](../../../../src/core/carnot_mobius_ledger.py#L25).
* `hook_cleanroom_sem_tech(tailings_count, saltwater_mb, energy_fe)`: Runs [SEMElectrochemicalExtractor](../../../../src/environment/cleanroom_mechanics.py#L523) closed-loop tailings leaching, precipitating Au, Ag, PGMs, Cu, Ni, and rare earths at 95% brine recirculation.
* `hook_cleanroom_aeronautics(balloon_volume_m3, gas_type, propeller_rpm, dt)`: Simulates [AeronauticContraption](../../../../src/environment/cleanroom_mechanics.py#L614) Archimedes lift, sleeve-valve combustion propeller thrust, and [DruckerPragerProjection](../../../../src/core/yield_criteria.py#L42) structural yield.
* `hook_cleanroom_nutrition(food_id, nutrition, saturation)`: Tracks [NutritionalDiversityTracker](../../../../src/environment/cleanroom_mechanics.py#L724) Carrot max HP progression milestones and Onion Shannon entropy diversity buffs/malaise penalties.
* `hook_cleanroom_conduits_rednet(channel_color, signal_strength, power_active, fluid_active)`: Manages [CompositeVoxelConduit](../../../../src/environment/cleanroom_mechanics.py#L867) multi-medium transport and [RedNet16BundledCable](../../../../src/environment/cleanroom_mechanics.py#L845) 16-channel analog signals.
* `hook_cleanroom_logistics(belt_tier, inserter_speed, stack_size, dt)`: Steps dual-lane [TransportBeltSegment](../../../../src/environment/cleanroom_mechanics.py#L889) items and [DirectionalInserter](../../../../src/environment/cleanroom_mechanics.py#L905) swing cycles.
* `hook_cleanroom_mob(base_hp, attack_damage, affix_1, affix_2, subtype, ambulatory_class)`: Applies [AdversaryEntityProfile / MobProperties](../../../../src/environment/cleanroom_mechanics.py#L346) affix invariants via [InfernalAffix](../../../../src/environment/cleanroom_mechanics.py#L318) composed with canonical [AdaptiveSkeletonHarness](../../../../src/core/adaptive_skeleton_harness.py#L27) morphology.
* `hook_cleanroom_fauna_genetics(mutation_rate)`: Breeds [FaunaGeneticsComponent](../../../../src/environment/cleanroom_mechanics.py#L920) specimens with Mendelian allele crossover and mutation.
* `hook_cleanroom_inventory(item_id, count)`: Inserts ItemStacks, FluidStacks, and GasStacks into [ExpandedInventorySystem](../../../../src/environment/cleanroom_mechanics.py#L116).
* `hook_create_rotational_network()`: Balances torque precession and stress capacity (SU) across [RotationalKineticNetwork](../../../../src/environment/cleanroom_mechanics.py#L1014).
* `hook_tconstruct_smeltery()`: Evaluates Mohr-Coulomb shear yield and geotechnical tool traits in [ModularTool](../../../../src/environment/cleanroom_mechanics.py#L1063) and [SmelterySystem](../../../../src/environment/cleanroom_mechanics.py#L1110).
* `hook_jade_raycast()`: Raycasts block coordinates through [JadeRaycastInspector](../../../../src/environment/cleanroom_mechanics.py#L1233) evaluating gyroidic spectral admissibility ($H_{\text{spec}}$).
* `hook_ferritecore_fastmap()`: Deduplicates blockstate bitvectors via [FastMapBlockState](../../../../src/environment/cleanroom_mechanics.py#L1279) carry-free XOR residues.
* `hook_waystones()`: Executes topological teleportation transitions through [WaystonesNetwork](../../../../src/environment/cleanroom_mechanics.py#L1329) RP4 projective boundaries.
* `hook_all_the_heads()`: Assesses cervical joint shear fracture on procedural rigs via [CranialJointShearRegistry](../../../../src/environment/cleanroom_mechanics.py#L1488).
* `hook_projecte_emc()`: Computes closed-loop transmutation values using the [ProjectEEMCSolver](../../../../src/environment/cleanroom_mechanics.py#L1673) Leontief input-output governor.
* `hook_bag_of_holding()`: Manages nested dimensional storage and void safeguards using [BagOfHolding](../../../../src/environment/cleanroom_mechanics.py#L1956).

#### Exterior Mechanics Hooks:
* `hook_chisel_octree(x, y, z, carve, material_id)`: Carves or places micro-voxel bits into the Morton-encoded [PointerlessOctree](../../../../src/ui/voxelboxter_simulation.py#L767) (Chisels & Bits).
* `hook_adaptive_rig(subtype, ambulatory_class, difficulty_scale)`: Synthesizes procedural inverse-kinematic skeletons via [AdaptiveSkeletonHarness](../../../../src/core/adaptive_skeleton_harness.py#L27).
* `hook_dual_yield_stress(pressure_val, shear_val, friction_angle, cohesion)`: Simulates geotechnical dual-regime plasticity via [MohrCoulombProjection](../../../../src/core/yield_criteria.py#L13) and [DruckerPragerProjection](../../../../src/core/yield_criteria.py#L42).
* `tick_vehicle_kinetics(dt)`: Simulates plasma air induction N2/O2 ionization and sleeve-valve combustion aerodynamics.
* `tick_life_and_hunger(dt, current_pressure, current_mischief)`: Evaluates [ValenceFunctional](../../../../src/core/valence_drive.py#L20) and [EgoDeathThresholdMonitor](../../../../src/core/archetype_engines.py#L30).
* `evaluate_economy_thermodynamics()`: Monitors topological waste heat and Carnot efficiency on the admin shop using [CarnotMobiusLedger](../../../../src/core/carnot_mobius_ledger.py#L25).


