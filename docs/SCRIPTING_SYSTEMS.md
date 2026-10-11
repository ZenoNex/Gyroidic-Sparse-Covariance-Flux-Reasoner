# Gyroidic Scripting Systems

This document explains the three distinct tiers of scripting available within the Gyroidic Sparse Covariance Flux Reasoner ecosystem. Rather than a monolithic scripting language, the system partitions control into **Semantic**, **Structural**, and **Physical** tiers, enforcing the philosophy of "Patch Sovereignty" across its biomimetic components.

---

## Tier 1: Semantic Scripting (Diegetic Terminal)

The **Diegetic Terminal** (`http://localhost:8000`) is the highest-level interface, driven by [`diegetic_terminal.html`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_terminal.html) and [`diegetic_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py). It provides control over the semantic meaning and cognitive "focus" of the Reasoner.

### Capabilities
- **Manifold Injection**: Users can inject raw semantic concepts, text, or multimodal dyad associations (images, audio).
- **Control Flags**: Adjust the Reasoner's regime (GOO vs. PRICKLES) and dyad commutativity.
- **Cognitive Routing**: By associating concepts, you script the "thought process" of the Reasoner as it navigates the embedding manifold.

*For full details on semantic routing, see [INTERFACE_LAYER.md](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/INTERFACE_LAYER.md).*

---

## Tier 2: Structural Scripting (Voxelboxter CLI & REPL)

**Voxelboxter** ([`voxelboxter_client.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_client.py)) acts as the tangible, PyBevy-based voxel client where abstract math turns into structural physics. It features a chat-based command line interface that provides Structural Scripting.

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

> [!NOTE]
> These commands use unrestricted `eval()` and `exec()`. This is by design. The system does not sandbox the Administrator; to restrict `locals` would limit the kind of mods that could be generated, breaking the integrity of the sovereign user.

---

## Tier 3: Physical Scripting (Node Environment)

The most direct, hardware-adjacent scripting layer is the **Physical Node Editor** ([`node_environment.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py)), built using `dearpygui`.

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
Nodes communicate through strictly typed sockets defined by [`SocketType`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L75):
* **Blue Sockets (`VECTOR`):** 3D positional/directional vectors `[X, Y, Z]`. Used for coordinate offsets, impulse vectors, and velocity vectors.
* **Yellow Sockets (`COLOR`):** 4-element color vectors `[R, G, B, A]` in normalized floating point or OKLab space.
* **Gray Sockets (`FLOAT`):** Single scalar values (e.g., hardness, friction, mass cost, battery charge).
* **Green / Pink Sockets (`INT` / `BOOLEAN`):** Discrete integer counting numbers (material IDs, iterations) and boolean flags.
* **Socket Topology (Diamond vs. Circle):**
  * **Circle Socket:** Carries a uniform scalar or vector value applied across an entire construct or component.
  * **Diamond Socket (Field):** Carries per-vertex or per-voxel spatial field calculations evaluated dynamically across the geometric manifold.

### 3. Master Node Groups & Game Engine Attribute Baking
To avoid rebuilding repetitive node graphs across modular component meshes, the environment provides **Master Node Groups** ([`MasterNodeGroup`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L207)):
* **Packaging and Reusability:** Collapses complex mathematical subgraphs into reusable macros with exposed sliders.
* **Center of Gravity (COG) Offset Logic:**
  * Reads the mesh dimensions from a `Bounding Box` calculation or explicit dimension vector.
  * Multiplies the dimension vector by `-0.5` (or negates the midpoint $(\text{min} + \text{max}) \cdot 0.5$) to calculate the exact geometric midpoint offset.
  * Offsets the geometry vertices relative to the origin so that physics pivots rest precisely dead center.
* **Named Vertex Attribute Injection:**
  * Bakes custom physical parameters (`phys_hardness`, `phys_friction`, `phys_hp`, `cog_offset`) into the vertex data layer prior to export.
  * Downstream physics engines and vehicle controllers read `phys_friction` for tire traction, `phys_hardness` for damage thresholds, and `cog_offset` for rigid body mass distribution.

### 4. Material Registries & Ore Dictionary
* **Resource Registry ([`ResourceRegistry`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L93)):** Catalogs material density ($\text{kg/m}^3$), Mohr-Coulomb cohesion yield hardness, energy density ($\text{MJ/kg}$), and base commercial credit valuation across standard substances (Stone, Iron Ore, Iron Ingot, Diamond, Redstone, Wood Log).
* **Ore Dictionary ([`OreDictionary`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L262)):** Unifies disparate block, ore, scrap, and ingot IDs under canonical tags (`oreIron`, `ingotIron`, `gemDiamond`, `dustRedstone`, `woodLog`, `chiselBitStone`, `chiselBitIron`, `carbideVoxel`, `darkMatterFlux`, `cellOxygen`, `bioRation`, `thrusterBlock`, `batteryBlock`), enabling polymorphic recipe resolution.

### 5. Direct Subsystem Integrations (Zero Redundant Wrappers)
The node environment directly interacts with canonical engine classes:

1. **Discrete Bit Carving ([`PointerlessOctree`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L767) & [`BooleanXORLayer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/structural_blueprints.py#L42)):**
   * Carves and places 1/16th scale sub-voxel bits (4096 micro-bits per macro block) via [`chisel_carve_bit`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L453) and [`chisel_place_bit`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L472).
   * Employs 3D Morton Z-order curve bit-interleaving for linear memory access and recycles carved debris into scrap inventories.
2. **Crafting & Addon Assembly ([`AddonRoutine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/structural_blueprints.py#L12) & [`InventoryComponent`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L52)):**
   * Multi-stage recipe assembly validating material costs and deducting mass from inventories via [`craft_addon_layer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L494).
3. **Vehicle Kinetics & Delta-v Collision Physics ([`RigidBody`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L104), [`AirBreathingBattery`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L173), [`VehicleEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L225)):**
   * **Real-Time Impulse Dynamics:** Computes dynamic velocity changes frame-by-frame via [`simulate_collision_impulse`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L530):
     $$\Delta v = \frac{J}{m}$$
     where $J$ is the collision impulse magnitude and $m$ is the vehicle mass.
   * **$\Delta t$ Hardness Curve:** Deceleration over a prolonged $\Delta t$ results in controlled crumpling; an identical $\Delta v$ over a near-zero $\Delta t$ causes catastrophic structural failure.
   * **Tiered Damage Thresholds:**
     * Low Tier ($\Delta v < 2.235\text{ m/s}$ / $5\text{ mph}$): Cosmetic scuffs and surface particle dust.
     * Mid Tier ($2.235\text{ m/s} \le \Delta v \le 6.705\text{ m/s}$ / $15\text{ mph}$): Panel denting and localized voxel detachment.
     * High Tier ($\Delta v > 6.705\text{ m/s}$ / $15\text{ mph}$): Catastrophic structural shearing and block ejection from inventory.
   * **Tuning Augments:**
     * `reinforced_alloy_chassis`: Doubles structural $\Delta v$ shearing thresholds.
     * `impact_energy_harvester`: Converts high-severity kinetic impact energy directly into temporary velocity and nitro boosts.
   * **Plasma Air Induction:** `AirBreathingBattery` features plasma ionization (`plasma_air_induction = True`, 2.4x boost factor) converting ambient atmospheric $\text{N}_2$ and $\text{O}_2$ into high-energy oxidizers, bypassing environmental $\text{NO}_2$ synthesis scarcity.
   * **Sleeve-Valve Combustion Timings:** `VehicleEngine` simulates sleeve-valve electrical advance timings (12.5 degrees) and port overlap efficiencies (1.35x) to optimize power draw and thrust curves in [`tick_vehicle_kinetics`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L580).
4. **Vitality & Life/Hunger Governor ([`ValenceFunctional`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/valence_drive.py#L20) & [`EgoDeathThresholdMonitor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/archetype_engines.py#L30)):**
   * Metabolic burn and hunger depletion scale dynamically across `DifficultyMode` (`PEACEFUL`, `SURVIVAL`, `HARSH`, `ENTROPIC`) in [`tick_life_and_hunger`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L675).
   * Evaluates dissonance and manifold hunger via `ValenceFunctional` and monitors non-ergodic abstraction rates ($R_a$) to safeguard against ego-death collapse.
5. **Procedural Character Rigs & Enemy Subtypes ([`AdaptiveSkeletonHarness`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/adaptive_skeleton_harness.py#L27)):**
   * Generates rigged procedural skeletal structures across `AmbulatoryClass` kinematics (bipedal, quadruped, hexapod, avian, serpentine, floating) in [`generate_enemy_rig`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L718).
   * Injects `EnemySubtype` deformations (e.g., `ABSTRACTED_GLITCH` phase noise, `DEMIURGIC_TITAN` localized $dt$ dilation, `VOID_STALKER` shadow geometry).
6. **Thermodynamic Admin Shop ([`CarnotMobiusLedger`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/carnot_mobius_ledger.py#L25) & [`LeontiefGovernor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/leontief_governor.py#L21)):**
   * Admin and visitor trading bounded by thermodynamic waste heat dissipation ($P_{\text{waste}}$) and Carnot efficiency limits ($\eta_{\text{stack}}$) via [`execute_admin_trade`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L733) and [`evaluate_economy_thermodynamics`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L761).

### 6. The Efficiency Singularity Integration
The node environment directly binds to the core architectural pillars of **The Efficiency Singularity** through [`hook_voxelboxter_efficiency_singularity`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L621):
* **Wager #1 (90/10 Rejection Ratio):** Non-scalar admissibility rejection probe evaluated in $O(1)$ via [`VoxelboxterAdmissibilityGate`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L380), filtering construct proposals before expensive topological integration.
* **Wager #2 (80/20 Fossilization Split):** Structural graph core partition into immutable bone ($80\%$) and plastic flesh ($20\%$) via `fossilize_core`, dramatically bounding memory churn during simulation updates.
* **Wager #3 (99/1 Structure/Weight Bet):** Topological macro-move surgeries ($\Delta \text{Topology} \ge 1$) executed across bounding planes via `execute_topological_macro_move`.
* **Wager #4 (Spectral Speculation):** Frequency-domain fastpass evaluated via [`SpectralSpeculationEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L425) using prime harmonic resonance ladders; constructs exhibiting low spectral entropy ($H_{\text{spec}} < \epsilon$) bypass heavy step solvers.
* **Non-Conservative Circulation:** Local neighborhood stress relaxation executed in $O(1)$ via `repair_local_neighborhood`.
* **Non-Ergodic Kelly Defense:** Growth factor governing complex structural dynamics:
  $$g_{\text{time}} = \mu - \frac{\sigma^2}{2}$$
  preventing catastrophic ruin under volatility spikes.

### 7. D-Wave Collective Computation Pool
To prevent UI blocking during heavy mathematical evaluations and DSP processing, the Physical Scripting tier uses an **asynchronous evaluation worker thread** ([`_evaluation_worker`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1910)):
* Evaluates node graphs in a dedicated side-chain thread while monitoring hardware headroom via [`has_headroom`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/hardware_monitor.py#L12).
* Tracks slider states (Latent Dimension, Resolution, Mass Cost, Topological Persistence, Quantum Tunneling Probability).
* Monitors trajectory stability through the [`CALM`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/surrogates/calm_predictor.py#L22) surrogate predictor, automatically aborting compiles if the entropic collapse score exceeds $0.8$.
* Balances network egalitarian consensus via [`BonfireNomadicRing`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/p2p/bonfire_consensus.py#L24) and [`FreenetClient`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/p2p/freenet_ws_client.py#L25).
* When a functional node links to the `Voxelboxter Graph Sink`, the worker dynamically compiles the layer ([`MangostienBSplineMod`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/structural_blueprints.py#L88), [`DarkMatterAttractorLayer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/structural_blueprints.py#L140)), signs it with honest silicon jitter via [`harvest_honest_jitter`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/honest_jitter.py#L15), validates Chern-Simons topological invariants via [`ZKAggregator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/p2p/zk_aggregator.py#L26), and hot-injects it directly into the running PyBevy `PatchStateResource.routine`.

---

## Tier 4: Cleanroom Systems Suite (Industrial, Ecological & Logistical)

The Physical Scripting and Voxelboxter engines incorporate cleanroom implementations of classic sandbox, automation, and ecological mechanics, bridging them directly into the underlying topological manifold:

### 1. Spatial Reconnaissance & Waypoints ([`JourneyTopoRadar`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L186))
* **Inspiration:** JourneyMap.
* **Architecture:** Projects 3D voxel heightmaps and cavern depths into 2D topographical radar slices.
* **Death Fossils:** When an entity abstracts or dies, a permanent historical stress tensor marker (`TopoWaypoint` with $\beta = 0$) is fossilized at the death coordinate, anchoring non-ergodic memory across reloads.
* **Node Interface:** `JourneyTopoRadar (JourneyMap)` node controls radar radius, subterranean slice filtering, and entity blip output vectors.

### 2. Resource Distributions & Drop Matrices ([`ResourceDistributionInspector`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L249))
* **Inspiration:** Just Enough Resources (JER).
* **Ore Height Distributions:** Models ore spawn probability density curves across Chebyshev depth bands $z \in [-64, 320]$ for coal, iron, copper, gold, redstone, diamond, uranium, and dark matter flux.
* **Mob Drops & Dungeon Manifolds:** Configures drop chances scaled by looting levels and non-ergodic entropy ($H_{\text{mischief}}$).
* **Node Interface:** `Resource Distribution (JER)` node queries height density and drop probabilities for arbitrary material IDs.

### 3. Procedural Mob Attributes & Elite Modifiers ([`MobProperties`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L346) & [`InfernalAffix`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L318))
* **Inspiration:** Mob Properties & AtomicStryker's Infernal Mobs / Multi Mine.
* **Attribute Control:** Configures max HP, movement speed, attack damage, knockback resistance, and follow range.
* **Infernal Affixes:** Elite mobs dynamically spawn with 1-4 random affixes (`1UP`, `Alchemist`, `Berserk`, `Blastoff`, `Bulwark`, `Cloaking`, `Darkness`, `Ender`, `Exhaust`, `Fiery`, `Ghastly`, `Hermetic`, `Lifesteal`, `Ninja`, `Poisonous`, `Quicksand`, `Regen`, `Rust`, `Sapper`, `Sprint`, `Storm`, `Twin`, `Webbing`, `Wither`).
* **Multi-Mine Persistent Memory:** [`MultiMineMemory`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L386) tracks block fracture damage across ticks and multi-agent mining attempts, preventing tool-release damage resets.
* **Node Interface:** `Adversary Morphology & Affix Invariants` node mutates entity baselines into elite adversaries.

### 4. Multi-Tier Metallurgical Dissolution ([`MekanismProcessingPipeline`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L451))
* **Inspiration:** Mekanism, Mekanism Tools, Mekanism Generators.
* **Tiered Multiplication:**
  * **Tier 1 (Smelting):** 1x yield (Ore $\to$ 1 Ingot).
  * **Tier 2 (Enrichment):** 2x yield (Ore $\to$ 2 Dust $\to$ 2 Ingots).
  * **Tier 3 (Purification):** 3x yield using Oxygen injection (Ore + $\text{O}_2 \to$ 3 Clumps $\to$ 3 Shards $\to$ 3 Dust).
  * **Tier 4 (Chemical Injection):** 4x yield using Hydrogen Chloride ($\text{HCl}$) gas.
  * **Tier 5 (Chemical Dissolution):** 5x yield using Sulfuric Acid ($\text{H}_2\text{SO}_4$) dissolution into clean slurry, followed by crystallizer precipitation.
* **Node Interface:** `Mekanism Ore Refinery (1x-5x)` node balances raw ore input against chemical reagent volumes to calculate refined metal yields.

### 5. Electrochemical Minerals Extraction ([`SEMElectrochemicalExtractor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L523))
* **Inspiration:** SEM TECH Saltwater Electrolyzer (Rowow Open-Source CMU).
* **Closed-Loop Tailings Leaching:** Consumes mineral tailings, saltwater brine, and electrical energy ($\text{FE}$) to electrodeposit precious metals (Au, Ag, PGMs), base metals (Cu, Ni), and critical rare earths.
* **Brine Recirculation:** Recovers $95\%$ of saltwater while monitoring ion-exchange membrane durability degradation.
* **Node Interface:** `SEM Tech Electrolyzer (Saltwater + Electricity)` node controls tailings throughput, saline volumes, and electricity feed.

### 6. Buoyant Airships & Propeller Flight ([`AeronauticContraption`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L614))
* **Inspiration:** Create: Aeronautics.
* **Buoyancy Physics:** Computes Archimedes buoyant force ($F_{\text{buoyant}} = (\rho_{\text{air}} - \rho_{\text{gas}}) \cdot V \cdot g$) for helium and hot-air envelopes against ambient `EnvironmentalAtmosphere`.
* **Aerodynamics:** Computes propeller thrust curves and aerodynamic drag ($F_{\text{drag}} = \frac{1}{2}\rho v^2 C_d A$) to govern rigid flying contraptions detached from the block grid.
* **Node Interface:** `Create: Aeronautics Contraption` node balances balloon volume, gas types, and RPM into net lift and thrust vectors.

### 7. Culinary Nutrition & Diet Progression ([`NutritionalDiversityTracker`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L724))
* **Inspiration:** Farmer's Delight & Spice of Life (Carrot & Onion Editions).
* **Carrot Edition (Milestone Heart Progression):** Discovering unique food items permanently increases max HP ($+2$ HP for every 5 unique foods discovered).
* **Onion Edition (Dynamic Rotational Buffs):** Evaluates Shannon diversity entropy over a rolling window of recent meals (12 meals). High diet diversity activates temporary buffs (`SPEED_II`, `HASTE_I`), while dietary monotony triggers nutritional malaise slowness.
* **Node Interface:** `Spice of Life & Nutrition` node tracks meal history, hunger saturation, and active status effects.

### 8. Multi-Bus Conduits & 16-Channel RedNet ([`CompositeVoxelConduit`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L867) & [`RedNet16BundledCable`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L845))
* **Inspiration:** EnderIO & RedNet / MineFactory Reloaded.
* **Single-Voxel Multiplexing:** Power, fluid, gas, item, and signal conduits coexist within a single voxel coordinate space without cross-talk or block collision.
* **RedNet Bundled Subnets:** Carries 16 distinct colored redstone subnets with individual analog signals ($0\text{--}255$) through a single bundled cable connection.
* **Node Interface:** `EnderIO & RedNet Bundled Cable` node configures channel colors and bus routing.

### 9. Pressurized Gasses, Scuba Systems, Tracer Life Support & Gauge Bridges
* **Inspiration:** Mekanism Pressurized Tubes, Scuba Gear, Rotary Condensentrators, Space Survival Life Support.
* **Pressurized Gas Conduits:** Supports [`GasTubeTier`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1987) (`BASIC`, `ADVANCED`, `ELITE`, `ULTIMATE`) with throughput scaling from $256\text{ mB/t}$ to $16384\text{ mB/t}$, paired with [`GasDumpingMode`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1995) (`IDLE`, `DUMPING`, `DUMPING_EXCESS`).
* **Scuba Breathing Apparatus ([`MekanismScubaTank`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L2015)):** Portable life support reservoir consuming compressed oxygen gas at $2.5\text{ mB/s}$ in vacuum or unbreathable biomes.
* **Rotary Condensentrator ([`RotaryCondensentrator`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L2036)):** Bidirectional thermodynamic phase-change machine converting pressurized gasses to fluids (condensentrating) and fluids to gasses (decondensentrating) at $5\text{ FE/mB}$.
* **Brick-Based Tracer Oxygen Engine ([`TracerOxygenAtmosphereEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L2081)):** Simulates discrete atmospheric molecule packets, obstacle raycast occlusions, and dynamic hull punctures triggering explosive depressurization.
* **$U(1)$ Gauge Field Atmospheric Bridge ([`U1GaugeAtmosphereBridge`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L2158)):** Formulates atmospheric pressure and flow as a relativistic $U(1)$ gauge 4-potential $A_\mu$, computing the electromagnetic-analogue field tensor $F_{\mu\nu}$, invariant $F^2$, Lagrangian action density $\mathcal{L} = -\frac{1}{4}F_{\mu\nu}F^{\mu\nu}$, and enforcing the continuity equation $\partial_\mu J^\mu = 0$ for hermetic breach detection.

### 10. Automated Logistics & Directional Inserters ([`TransportBeltSegment`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L889) & [`DirectionalInserter`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L905))
* **Inspiration:** Factorio & BuildCraft.
* **Dual-Lane Belts:** Models transport belts with two independent item lanes, supporting tiers from $15\text{ items/s}$ up to $45\text{ items/s}$.
* **Directional Inserters:** Simulates pickup, swing timing, filter selection, and stack capacities between inventories and transport belts.
* **Node Interface:** `Factorio Logistics Network` node regulates logistics throughput across automated assembly lines.

### 11. Mendelian Animal Husbandry ([`FaunaGeneticsComponent`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L920))
* **Inspiration:** Animal Husbandry & Genetics mods.
* **Allelic Inheritance:** Chromosomal alleles for movement speed, jump height, and material yields, supporting crossover and mutation rates influenced by ambient gyroidic flux.
* **Node Interface:** `Animal Husbandry & Genetics` node monitors expressed speeds, jumps, and yields across generations.

### 12. Rotational Kinetic Networks ([`RotationalKineticNetwork`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1014))
* **Inspiration:** Create Mod.
* **Stress Capacity vs. Impact:** Tracks RPM, torque directions, stress capacity ($SU$), and stress impact across kinetic shafts, waterwheels, and crushing wheels, halting rotation upon network overstress.

### 13. Modular Tool Forging & Smeltery Casting ([`ModularTool`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1063) & [`SmelterySystem`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1110))
* **Inspiration:** Tinkers' Construct.
* **Component Assembly:** Tool heads, handles, and bindings forged from varied materials (Cobalt, Ardite, Wood, Manyullyn) contributing material traits (`Lightweight`, `Ecological`, `Writable`) and modifiers (Redstone haste, Quartz sharpness).
* **Smeltery Alloying:** Melts ores into molten fluid tanks and casts them into casts (Pickaxe Head, Ingot, Gear) with stoichiometric alloy fusing (e.g. Copper + Tin $\to$ Bronze).

### 14. Rustic Brewing & Cotton Processing ([`RusticDelightManager`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1192) & [`FermentationBarrel`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1211))
* **Inspiration:** Rustic & Farmer's Delight additions.
* **Coffee Roasting & Brewing:** Roasts raw coffee cherries and brews restorative beverages providing haste and speed status effects.
* **Cotton Ginning:** Separates raw harvested cotton into structural string fibers and replantable seeds.

### 15. Raycast Inspection HUD ([`JadeRaycastInspector`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1233))
* **Inspiration:** Jade, WAILA, HWYLA.
* **Real-Time Voxel Query:** Raycasts target block IDs to return harvest tool requirements, harvest level ratings, break progress, crop maturity percentage, and spectral admissibility.

### 16. Blockstate Deduplication ([`FastMapBlockState`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1279))
* **Inspiration:** FerriteCore.
* **FastMap Interning:** Replaces bulky property hash maps with canonical integer state IDs and packs 6-direction neighbor occlusion bitmasks into single bytes.

### 17. Dimensional Waystone Network ([`WaystonesNetwork`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1329))
* **Inspiration:** Waystones.
* **Teleportation Anchors:** Inter-dimensional and cross-biome travel calculating non-linear distance curves and player XP level costs.

### 18. Instanced Container Loot ([`LootrContainerManager`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1412))
* **Inspiration:** Lootr.
* **Per-Player Dungeon Chests:** Generates independent, non-depleting loot instances for each player visiting dungeon and ruin containers.

### 19. Anvil Repair Economy ([`EasyAnvilsSystem`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1455))
* **Inspiration:** Easy Anvils.
* **Prior Work Penalty Elimination:** Computes transparent repair costs based purely on material counts and enchantment ranks, removing cumulative exponential repair lockouts.

### 20. Cranial Joint Shear & Trophies ([`CranialJointShearRegistry`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1488))
* **Inspiration:** All The Heads.
* **Biomechanical Rupture:** High-impulse critical strikes exceeding cervical shear yield drop decorative and collectible mob trophy fossils.

### 21. Fluid XP Printing & Hyper-Enchanting ([`EnchantmentIndustryPipeline`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1542))
* **Inspiration:** Create: Enchantment Industry.
* **Printing Presses:** Consumes liquid experience to duplicate enchanted books and push enchantments past normal survival caps into hyper-enchanted tiers (e.g., Efficiency VI+).

### 22. Biome Hydrologic Flow ([`ClimateRiverSegment`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1600))
* **Inspiration:** Climate Rivers.
* **Vector Currents:** Assigns directional velocity vectors and channel widths to waterways based on biome terrain slope gradients (alpine rapids vs. meandering marsh).

### 23. Modern Combat Mechanics ([`CombatNouveauProfile`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1625))
* **Inspiration:** Combat Nouveau (Jeb Combat Test 8c).
* **Attack Reach & Sweep:** Differentiates attack reach distances, weapon charge recovery curves, sweeping edge angles, and sprint-interruption mechanics across swords, axes, and polearms.

### 24. Hotbar Inventory Swapping ([`HotbarSwapper`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1650))
* **Inspiration:** Hotbar Swapper / Hotbar Keybinds.
* **Paging Matrix:** Instantly swaps the active 9-slot hotbar with any inventory row via single keybind signals.

### 25. Energy-Mass Equivalence ([`ProjectEEMCSolver`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1673))
* **Inspiration:** ProjectE / Equivalent Exchange 2.
* **Recursive EMC Solver:** Assigns immutable energy-mass values to base elements and computes compound item valuations via Leontief graph recursion, supporting item dissolution into raw EMC and transmutation back into discovered items.

### 26. Diegetic Alchemy & Cauldron Brewing ([`DiegeticCauldron`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1739))
* **Inspiration:** Witchery / Inspirations Cauldron Brewing.
* **Thermodynamic Boiling:** Simulates cauldron fluid temperature, rotational stirring direction (clockwise vs. counter-clockwise), and sequential reagent chemistry to brew distinct potion profiles.

### 27. Multipart Micro-Cells & Subgrid Harnesses ([`MultipartVoxelCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1878) & [`SubgridRotationalWireHarness`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1906))
* **Inspiration:** Forge Multipart & Project Red.
* **Sub-Voxel Partitions:** Multiple functional parts (vertical slabs, edge covers, conduit faces) share a single voxel cell without collision clipping, with subgrid wire rotational transforms mapping internal coordinates into world space.

### 28. Dimensional Pocket Storage ([`BagOfHolding`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1956))
* **Inspiration:** Extra Utilities & Bag of Holding.
* **Tiered Expansion & Void Collapse:** Expands inventory capacity across tiers (Leather, Iron, Gold, Diamond) while enforcing a strict singularity safeguard: nesting a bag within another triggers catastrophic void collapse.

### 29. Crop Breeding & Giant Multi-Blocks ([`IC2CropGenome`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1925) & [`GiantCropCluster`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1940))
* **Inspiration:** IndustrialCraft 2 Crops & Giant Crop Multi-Blocks.
* **Genomic Stats:** Tracks `Growth`, `Gain`, and `Resistance` stats through cross-pollination breeding matrices, with mature 3x3 grids fusing into dense giant harvest clusters.

---

## Master Execution Hook Registry (`PhysicalNodeEditor`)

The DearPyGui [`PhysicalNodeEditor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L316) exposes a comprehensive suite of Python hooks executed headlessly or through the DPG interface:

### Canonical Subsystem & Physics Hooks
| Method Link | Parameters | Description |
| :--- | :--- | :--- |
| [`chisel_carve_bit`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L453) | `x, y, z, material_id` | Carves micro-voxel bits into the Morton-encoded [`PointerlessOctree`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L767) and deposits scraps. |
| [`chisel_place_bit`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L472) | `x, y, z, material_id` | Inserts micro-voxel bits into the [`PointerlessOctree`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L767). |
| [`craft_addon_layer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L494) | `layer, layer_name, material_id, quantity` | Crafts blueprint construct layers using canonical [`AddonRoutine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/structural_blueprints.py#L12) and mass deduction. |
| [`simulate_collision_impulse`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L530) | `impulse_magnitude, contact_normal, delta_t` | Evaluates Delta-v collision severity tiers (`LOW`, `MID`, `HIGH`) and detaches voxels. |
| [`tick_vehicle_kinetics`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L580) | `dt` | Simulates vehicle aerodynamics with Plasma Air Induction and sleeve-valve combustion. |
| [`hook_voxelboxter_efficiency_singularity`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L621) | `candidate_cells, bone_ratio` | Binds Voxelboxter to the 4 Wagers, Non-Conservative repair, and Non-Ergodic Kelly growth. |
| [`tick_life_and_hunger`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L675) | `dt, current_pressure, current_mischief` | Simulates metabolic burn, starvation damage, [`ValenceFunctional`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/valence_drive.py#L20), and ego-death abstraction rates. |
| [`generate_enemy_rig`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L718) | `subtype, ambulatory_class, difficulty_scale` | Generates inverse-kinematic character rigs via [`AdaptiveSkeletonHarness`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/adaptive_skeleton_harness.py#L27). |
| [`execute_admin_trade`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L733) | `material_id, quantity, is_buy, role` | Executes shop clearance transactions with Kelly-bounded admin margins. |
| [`evaluate_economy_thermodynamics`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L761) | *(none)* | Evaluates market friction, Carnot efficiency ($\eta_{\text{stack}}$), and thermal runaway. |
| [`hook_dual_yield_stress`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1227) | `pressure_val, shear_val, friction_angle, cohesion` | Evaluates Mohr-Coulomb and Drucker-Prager geotechnical dual-regime plasticity flow. |

### Cleanroom Systems Integration Hooks
| Method Link | Parameters | Description |
| :--- | :--- | :--- |
| [`hook_cleanroom_radar`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L775) | `radar_radius, show_death_fossils, center_pos` | Scans entity blips and preserves permanent death waypoints via [`JourneyTopoRadar`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L186). |
| [`hook_cleanroom_jer`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L804) | `material_id, y_height, looting_level` | Evaluates Chebyshev ore density curves and drops via [`ResourceDistributionInspector`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L249). |
| [`hook_cleanroom_multi_mine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L825) | `coord, tool_power` | Progressively fractures blocks across ticks via [`MultiMineMemory`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L386). |
| [`hook_cleanroom_mekanism`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L849) | `tier, raw_ore, chemical_reagent_mb` | Executes 1x-5x metallurgical refining via [`MekanismProcessingPipeline`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L451). |
| [`hook_cleanroom_sem_tech`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L885) | `tailings_count, saltwater_mb, energy_fe` | Electrodeposits critical minerals from tailings via [`SEMElectrochemicalExtractor`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L523). |
| [`hook_cleanroom_aeronautics`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L911) | `balloon_volume_m3, gas_type, propeller_rpm, dt` | Computes lift, thrust, and yield via [`AeronauticContraption`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L614). |
| [`hook_cleanroom_nutrition`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L928) | `food_id, nutrition, saturation` | Computes Carrot max HP and Onion Shannon diversity buffs via [`NutritionalDiversityTracker`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L724). |
| [`hook_cleanroom_conduits_rednet`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L950) | `channel_color, signal_strength, power_active, fluid_active` | Routes power/fluids in [`CompositeVoxelConduit`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L867) and 16-color [`RedNet16BundledCable`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L845). |
| [`hook_cleanroom_gasses`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L979) | `gas_id, gas_amount_mb, tube_tier, dumping_mode, simulate_tracer, trigger_hull_leak` | Pressurized tubes, scuba breathing, rotary condensentrators, tracer oxygen leaks, and $U(1)$ gauge bridges. |
| [`hook_cleanroom_logistics`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1086) | `belt_tier, inserter_speed, stack_size, dt` | Simulates dual-lane [`TransportBeltSegment`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L889) and [`DirectionalInserter`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L905). |
| [`hook_cleanroom_mob`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1110) | `base_hp, attack_damage, affix_1, affix_2, subtype, ambulatory_class` | Composes Infernal Affixes with [`AdaptiveSkeletonHarness`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/adaptive_skeleton_harness.py#L27) morphology. |
| [`hook_cleanroom_fauna_genetics`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1145) | `mutation_rate` | Breeds offspring with allelic inheritance via [`FaunaGeneticsComponent`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L920). |
| [`hook_cleanroom_inventory`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1170) | `item_id, count` | Inserts items, fluids, and gasses into [`ExpandedInventorySystem`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L116). |
| [`hook_create_rotational_network`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1253) | `capacity_su, stress_su, rpm` | Computes kinetic speed, direction, and overstress in [`RotationalKineticNetwork`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1014). |
| [`hook_tconstruct_smeltery`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1267) | `tool_type, head_material, handle_material, redstone_mod, quartz_mod` | Forges modular tools and casts alloys in [`SmelterySystem`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1110). |
| [`hook_rustic_delight`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1303) | `raw_beans, hot_water_mb, raw_cotton` | Roasts coffee and gins cotton via [`RusticDelightManager`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1192). |
| [`hook_jade_raycast`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1320) | `block_id, block_name, crop_age` | Raycasts block properties and harvest viability via [`JadeRaycastInspector`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1233). |
| [`hook_ferritecore_fastmap`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1337) | `property_key, property_val` | Interns blockstates and bitpacks neighbor occlusion in [`FastMapBlockState`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1279). |
| [`hook_waystones`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1352) | `action, current_pos, target_name, xp_level` | Teleports players across dimensional waypoints in [`WaystonesNetwork`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1329). |
| [`hook_lootr_containers`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1370) | `container_id, player_uuid, tier` | Instantiates per-player loot containers via [`LootrContainerManager`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1412). |
| [`hook_easy_anvils`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1385) | `current_dura, max_dura, ingots, enchants` | Computes prior-penalty-free repair costs via [`EasyAnvilsSystem`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1455). |
| [`hook_all_the_heads`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1401) | `subtype, ambulatory, impact_impulse_J, beheading_lvl` | Rolls cervical joint shear decapitation via [`CranialJointShearRegistry`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1488). |
| [`hook_enchantment_industry`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1423) | `blank_books, book_name, fluid_xp_mb` | Prints books and hyper-enchants via [`EnchantmentIndustryPipeline`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1542). |
| [`hook_climate_rivers`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1441) | `biome_type, slope` | Computes biome velocity flow vectors via [`ClimateRiverSegment`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1600). |
| [`hook_combat_nouveau`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1458) | `category_str, charge_ratio, is_crit` | Computes attack damage and sweep reach via [`CombatNouveauProfile`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1625). |
| [`hook_hotbar_swapper`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1470) | `swap_row` | Swaps active hotbar slots across inventory rows in [`HotbarSwapper`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1650). |
| [`hook_projecte_emc`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1481) | `action, item_id, count, player_uuid` | Burns and transmutes items via [`ProjectEEMCSolver`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1673). |
| [`hook_cauldron_brewing`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1506) | `reagent, stir_dir, heat_active` | Heats, stirs, and brews potions in [`DiegeticCauldron`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1739). |
| [`hook_multipart_subgrid`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1524) | `side, subgrid_yaw` | Places vertical slabs and rotates subgrid wires via [`MultipartVoxelCell`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1878). |
| [`hook_bag_of_holding`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1540) | `is_nested_bag` | Inserts items and checks void collapse singularity in [`BagOfHolding`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1956). |
| [`hook_crop_breeding_giant`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1554) | `action, crop_species, g1, g2` | Cross-breeds crop alleles and checks 3x3 multi-block fusion via [`IC2CropGenome`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py#L1925). |
| [`evaluate_tick`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1581) | *(all simulation parameters)* | Comprehensive tick executing all 32+ hooks concurrently in headless or UI modes. |

---

## Interactive DearPyGui Node Reference

The GUI interface registered in [`_setup_nodes`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py#L1720) exposes 21 tangible visual nodes:

1. **`BSpline Mod Generator` (`node_bspline`):** Emits B-Spline latent tensor vectors parameterized by `Latent Dim` and `Resolution`.
2. **`Dark Matter Attractor` (`node_dark_matter`):** Emits dark matter fossil layers parameterized by `Iterations` and `Time Step`.
3. **`Global Physical Properties` (`node_physics`):** Global settings for `Mass Cost Mod`, `Topological Persistence`, `Resonance Frequency (Hz)`, and `Quantum Tunnel Prob`.
4. **`Master Node Group: Attribute Injector` (`node_master_group`):** Takes geometry vectors, accepts `phys_hardness` and `phys_friction`, and outputs baked attribute dictionaries with auto-calculated COG offsets.
5. **`Vehicle Subsystems & Flight` (`node_vehicle_components`):** Engine throttle, sleeve timing degrees, plasma air induction, reinforced chassis, and kinetic energy harvester toggles, outputting velocity vectors and impulse telemetry.
6. **`Character Rig & Enemy Subtypes` (`node_character_enemy`):** Procedural skeletal generation across `Ambulatory Class` and `Enemy Subtype` with difficulty scaling.
7. **`Admin Shop & Economy Sim` (`node_economy_shop`):** Role-based transaction clearance and admin margin modulation.
8. **`Voxelboxter Graph Sink` (`node_sink`):** The terminal compile sink that hot-injects completed mathematical mod layers into the live engine.
9. **`JourneyTopoRadar (JourneyMap)` (`node_journeymap`):** Radar radius and death fossil tracking.
10. **`Resource Distribution (JER)` (`node_jer`):** Chebyshev altitude band query and looting level modifiers.
11. **`Mekanism Ore Refinery (1x-5x)` (`node_mekanism`):** Processing tier (1-5), raw ore batches, and chemical reagents.
12. **`SEM Tech Electrolyzer (Saltwater + Electricity)` (`node_sem_tech`):** Tailings gangue stockpile, saltwater saline feed, electricity input, and membrane durability.
13. **`Create: Aeronautics Contraption` (`node_aeronautics`):** Balloon volume ($m^3$), buoyant gas (helium / hot air), and propeller RPM.
14. **`Spice of Life & Nutrition` (`node_nutrition`):** Food item ID, nutrition, saturation, and rolling Shannon entropy.
15. **`EnderIO & RedNet Bundled Cable` (`node_conduit_rednet`):** 16-color channel selector, analog signal ($0\text{--}255$), and power/fluid conduit toggles.
16. **`Factorio Logistics Network` (`node_factorio`):** Transport belt tier, inserter swing speed, and stack size.
17. **`Adversary Morphology & Affix Invariants` (`node_infernal_mobs`):** Base HP, attack damage, and 2 Infernal Affixes composed on procedural rigs.
18. **`Multi-Mine Progressive Fracture` (`node_multi_mine`):** Mining tool power modulating fracture damage accumulation.
19. **`Animal Husbandry & Genetics` (`node_fauna_genetics`):** Mutation rate modulating Mendelian inheritance.
20. **`Chisels & Bits Micro-Octree` (`node_chisels_octree`):** 3D bit coordinates ($X, Y, Z \in [0, 15]$) for sub-voxel placement and carving.
21. **`Geotechnical Dual-Yield (MC & DP)` (`node_dual_yield`):** Hydrostatic pressure and shear stress sliders outputting Mohr-Coulomb and Drucker-Prager plastic flow limits.

