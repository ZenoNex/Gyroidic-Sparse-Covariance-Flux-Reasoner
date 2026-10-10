# Voxelboxter Non-Dual Ecological Architecture

**Canonical References:**
- [voxelboxter_simulation.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py)
- [voxelboxter_client.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_client.py)
- [structural_blueprints.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/structural_blueprints.py)
- [yield_criteria.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/yield_criteria.py)
- [erosion_filter.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/erosion_filter.py)
- [garden_statistical_attractors.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/garden_statistical_attractors.py)
- [knowledge_dyad_fossilizer.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/knowledge_dyad_fossilizer.py)

---

## 1. Architectural Thesis: The Non-Dual Manifold

In this architecture, **Voxelboxter is not an isolated game engine wrapped in a superficial clean-room theme; it is the physical, diegetic manifestation of the exact same topological manifold that drives AI reasoning**.

There is no artificial barrier separating cognitive inference from physical simulation. The exact same mathematical primitives that calculate the system's thoughts:
- B-spline curves compiled via Kolmogorov-Arnold (KAN) networks
- Co-prime residue channels governed by Chinese Remainder Theorem (CRT) dynamics
- Prime resonance ladders and spectral eigenvalues
- Dual-scale Mohr-Coulomb and Drucker-Prager yield stress tensors

are the exact equations that physically generate the voxel terrain, simulate vehicle rigid body dynamics, model atmospheric oxidation in Air-Breathing Batteries (ABEB), carve weather and traffic gullies, and animate living flora and fauna ecologies in real time.

---

## 2. Ingestion as Topological Distillation, Not Clean-Room Isolation

When external signals (3D meshes, audio streams, text documents, or video frames) enter the system through `UniversalTopologyConverter`, `IVSTEncoder`, or `SovereignIngestor`, the objective is **Topological Distillation**:

1. **Decoupling Surface Expression from Structural Topology**:
   - Visual and acoustic media pass through Hann-windowed frame energy evaluations and Chebyshev $T_k$ recurrence polynomials.
   - Raw pixels, protected Euclidean coordinates, and audio samples are discarded.
   - The engine retains only a 96-coefficient visual fingerprint (32 modes across L/Cr/Cb channels) or a 64-coefficient audio harmonic vector.
   - High-frequency Chebyshev coefficients capture spectral texture density and invariant geometry rather than copyrightable surface expression.

2. **Topological Invariants & CRT Signatures**:
   - The `IVSTEncoder` projects byte-frequency histograms through polynomial bases into phase space.
   - It computes Betti numbers ($\beta_0, \beta_1$), surface-area-to-volume ratios, intrinsic graph volume, and Chinese Remainder Theorem (CRT) residue pressure signatures using dynamic prime ladders.

3. **Immediate Diegetic Population**:
   - Distilled topological signatures do not sit behind glass or inside cold databases.
   - They immediately populate the **Resonance Cavity** and the **Garden of Statistical Attractors** as active gravity wells that directly influence Voxelboxter's procedural physics and terrain formation.

---

## 3. Topological Algebra Directly Inhabiting Voxelboxter

Abstract topological streams translate directly into concrete, interactive voxel constructs:

### A. Non-Heuristic B-Spline Geometry (`BSplineCompiledMod`)
Custom vehicle chassis, hulls, mechanical tools, and terrain patches do not rely on static polygon meshes. They are compiled into exact, non-heuristic mathematical curves on the fly using Kolmogorov-Arnold (KAN) layers running the Cox-de Boor algorithm ([BSplineCompiledMod](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L309-L330)).
The B-splines are the ingested topological algebra made tangible.

### B. Morton Z-Order Octree Encoding (`morton_encode`)
To eliminate cache misses and GPU bus stalls during real-time ray-marching and terrain fracturing:
- 3D spatial coordinates $(x, y, z)$ are bit-interleaved into contiguous 1D Morton scalar indices via [morton_encode](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L333-L347).
- Lookups in [PointerlessOctree](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L760-L800) collapse into streaming memory, allowing `SiliconSovereigntyEngine` PyOpenCL hardware queues to execute zero-copy Wasserstein collapse directly over the active topological state.

### C. Dual-Scale Plasticity (Mohr-Coulomb vs. Drucker-Prager)
Vehicle collisions, tool chiseling, and projectile impacts interact with the voxel world through dual yield criteria ([apply_dual_yield_fracture](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L946-L985)):
1. **Mohr-Coulomb (Local Shear Yield)**:
   $$\tau = c + \sigma \tan\phi$$
   Governs local, sharp, brittle yield planes. When a vehicle takes a sharp turn, detonates a charge, or cuts a block, stress exceeding the shear limit breaks blocks along crisp rupture lines, leaving tire tracks, craters, and harvestable debris.
2. **Drucker-Prager (Global Flow Envelope)**:
   $$\alpha I_1 + \sqrt{J_2} - k = 0$$
   Wraps local rupture sites in a smooth, convex yield envelope. Global vehicle momentum and terrain contour flow smoothly around fractured zones without non-manifold mesh crashes.

---

## 4. Living Flora: The Cashew Tree of Pirangi Mutation

Voxelboxter flora incorporates botanical mutations modeled after the **Cashew Tree of Pirangi** (*Maior cajueiro do mundo* in Rio Grande do Norte, Brazil).

### Biological and Topological Phenomena
In normal trees, apical dominance and negative gravitropism cause branches to grow vertically upwards. In the Pirangi cashew tree, two genetic mutations produce a single organism spanning over 8,500 square meters (the size of 70 normal trees):

1. **Inverted Gravitropism**:
   Branches grow outward horizontally across the $(x, y)$ plane rather than ascending vertically.
2. **Cantilever Droop**:
   As a horizontal branch extends in length $L$ and accumulates wood mass $m$, gravitational load deflects its cantilever structure downward:
   $$\Delta z_{\text{droop}} = -\kappa \cdot m \cdot L^{1.3}$$
3. **Adventitious Ground Rooting**:
   When the drooping branch coordinate reaches the soil surface ($z \le z_{\text{ground}}$), contact does not terminate or rot the branch. Instead, the branch forms an [AdventitiousRootNode](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L380-L390) that penetrates the ground substrate.
4. **Secondary Trunk Metamorphosis**:
   The anchored root node metamorphoses into a secondary vertical trunk. This secondary trunk initiates its own radial horizontal branches, recursively repeating the cycle.
5. **Continuous Single-Organism Expansion**:
   The sprawling canopy is represented as a single connected [StructuralGraph](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L244-L295). Evaluating `graph.find_disconnected_components()` confirms that the entire forest-sized canopy remains a single continuous organism.
6. **Strict Mass Conservation**:
   Every block of wood, foliage, or root placed during [grow_tick](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L463-L590) is strictly deducted from the player or world [InventoryComponent](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L27-L31).

---

## 5. Living Fauna: Inhibition-Stabilized Networks (ISN)

Fauna populations operate through [FaunaISN](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L644-L710) and [FaunaComponent](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L630-L642):

1. **Excitation/Inhibition (E/I) Balance**:
   Cross-homeostatic coupling maintains $E/I$ balance:
   $$\tau_E \frac{dE}{dt} = -E + [W_{EE} E - W_{EI} I + I_{\text{ext}}]_+$$
   $$\tau_I \frac{dI}{dt} = -I + [W_{IE} E - W_{II} I + 0.5 I_{\text{ext}}]_+$$
   This prevents runaway panic swarming (epileptic explosion) or quiescent behavioral extinction across animal herds.
2. **Defect Propagation Partial Differential Equation (PDE)**:
   Environmental disruptions or habitat damage diffuse through the animal landscape via:
   $$\frac{\partial d}{\partial t} = D \nabla^2 d + \alpha V_{\text{gyroid}} - \beta d$$
   where $d$ is local environmental damage density, $D$ is spatial diffusion rate, $V_{\text{gyroid}}$ is the gyroidic manifold potential, and $\beta$ is recovery rate. Disturbances act like cytokine signals, coordinating collective ecological adaptation without global loss optimization.
3. **Chiral Gating**:
   Fauna steering trajectories are routed through left- or right-handed phase paths depending on local chirality $\text{sign}(E - I)$.

---

## 6. Weathering, Ley Lines, & Emergent Baking

1. **Weathering & Ley Line Gullies (`TopologicalErosionFBM`)**:
   Rain, wind, and heavy vehicle traffic apply Fractional Anisotropic Fractal Polynomial Functionals encoded Brownian Motion ([carve_weathering_ley_lines](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L987-L1010)). Multi-scale gullies are carved along pressure gradients $\nabla P$ using resonant prime frequencies. Gullies carved by heavy vehicles become **Resonance Streamlines (Ley Lines)** that lighter vehicles can physically slipstream through.

2. **Emergent Baking Engine (`EmergentBakingEngine`)**:
   Driven by `GardenStatisticalAttractors`, players and builders can "bake" localized geometry (homes, crop terraces, cooked provisions, flora groves) via user-triggered fossilization ([bake_construct](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L720-L755)). This converts subjective structural attachment into permanent topological persistence on the Poincaré disk.

## 7. Cleanroom Industrial, Logistical, and Ecological Synthesis

In harmony with the Non-Dual Manifold, classic gameplay systems from renowned sandbox mods and simulation games are cleanroom-engineered directly into Voxelboxter's topological physics rather than existing as isolated add-ons:

### A. Topographical Reconnaissance & Death Fossils (`JourneyTopoRadar` - JourneyMap Cleanroom)
- Real-time 2D/3D topological radar unrolls voxel slices, cavern conduits, and entity blips.
- When an entity abstracts or falls, its death coordinate is fossilized as a zero-Betti stress tensor (`TopoWaypoint`). This anchors historical trauma and structural memory onto the world map across reloads.

### B. Chebyshev Resource Depths & Loot Manifolds (`ResourceDistributionInspector` - JER Cleanroom)
- Ore distributions do not follow arbitrary random heights; they are modeled along Chebyshev polynomial density curves across depths $z \in [-64, 320]$ for coal, iron, copper, gold, redstone, diamond, uranium, and dark matter flux.
- Entity drops and fossil vault dungeon loot tables are dynamically weighted by Phase Alignment Scores ($PAS_h$) and non-ergodic mischief ($H_{\text{mischief}}$).

### C. Mutated Mob Attributes & Diablo Affixes (`MobPropertiesMutator` & `InfernalAffixRegistry` - Mob Properties & AtomicStryker Cleanroom)
- Entity parameters (health, movement speed, attack damage, follow range, equipment) are dynamically mutated by regional containment pressure.
- Elite adversaries manifest 1-4 random Infernal Affixes (`1UP`, `Berserk`, `Bulwark`, `Lifesteal`, `Storm`, `Webbing`, `Alchemist`, `Rust`, etc.), injecting tactical friction into combat.
- `MultiMineMemory` preserves block fracture strain across mining ticks and multi-agent operations, ensuring cooperative excavation is never wiped out by tool swings.

### D. Tiered Metallurgical Dissolution (`MekanismProcessingPipeline` - Mekanism Cleanroom)
- Ore processing scales across five physical-chemical tiers:
  1. **Tier 1 (Smelting):** 1x yield.
  2. **Tier 2 (Enrichment):** 2x yield via mechanical pulverization.
  3. **Tier 3 (Purification):** 3x yield using Oxygen gas oxidation into clumps and shards.
  4. **Tier 4 (Chemical Injection):** 4x yield using Hydrogen Chloride ($\text{HCl}$) gas.
  5. **Tier 5 (Chemical Dissolution):** 5x yield using Sulfuric Acid ($\text{H}_2\text{SO}_4$) slurry dissolution and crystallizer growth.

### D.1 Closed-Loop Saltwater Electrolysis & Tailings Recovery (`SEMElectrochemicalExtractor` - SEM TECH Cleanroom)
- **Early-Game Tech Progression Decoupling**: Solves the early-game technological bottleneck where high-value precious metals and rare earths traditionally demand late-game chemical infrastructure.
- **Feedstock**: Consumes accumulated low-grade **mine tailings, crushed rock gangue, and slag stockpiles** that typically clutter early-game storage.
- **Electrochemical Lixiviation**: Inspired by Rowow LLC / Robert Karas open-source hardware (CERN-OHL-S v2), a divided electrolysis cell with an ion-exchange membrane uses ambient **Saltwater (saline brine) + Electricity (FE/RF)** to generate nascent in-situ oxidizers (active chlorine/hypochlorite/dilute acid).
- **Direct Cathode Electrodeposition**: Dissolved metal ions migrate and reduce directly at the cathode as solid precipitates, powders, and foils (gold, silver, platinum group metals, copper, nickel, and rare earths).
- **Closed-Loop Sustainability**: Re-circulates $>95\%$ of the saline electrolyte without generating hazardous acid effluent, providing an accessible, high-yield metallurgical route before heavy sulfuric acid automation is constructed.

### E. Buoyant Aerodynamics & Airship Contraptions (`AeronauticContraption` - Create: Aeronautics Cleanroom)
- Detached rigid assemblies calculate Archimedean buoyancy ($F_{\text{buoyant}} = (\rho_{\text{air}} - \rho_{\text{gas}}) V g$) using helium or hot-air envelopes against ambient `EnvironmentalAtmosphere`.
- Converts sleeve-valve engine torque into propeller aerodynamic thrust and gyroscopic angular stability.

### F. Culinary Nutrition & Diet Progression (`NutritionalDiversityTracker` - Farmer's Delight & Spice of Life Cleanroom)
- **Carrot Milestone Progression:** Eating unique discovered food varieties permanently increases maximum health ($+2$ HP per 5 unique foods).
- **Onion Dynamic Diversity:** Evaluates Shannon entropy over a rolling 12-meal history; diverse diets grant kinetic buffs (`SPEED_II`, `HASTE_I`), while monotonous diets trigger metabolic malaise.

### G. Bundled Conduits & 16-Color Subnets (`CompositeVoxelConduit` & `RedNet16BundledCable` - EnderIO & RedNet Cleanroom)
- Single-voxel multiplexing allows power conduits, fluid pipes, item tubes, and redstone signals to coexist within a single grid voxel.
- 16 independent analog subnet channels (0-255) run through a single bundled cable without cross-talk.

### H. Dual-Lane Logistics & Automated Inserters (`LogisticsTransportNetwork` - Factorio Cleanroom)
- Transport belts feature two independent item lanes moving 15 to 45 items per second.
- Directional inserters execute pickup, swing timing, and filter sorting to automate assembly machines.

### I. Mendelian Animal Genetics (`FaunaGeneticsComponent` - Animal Husbandry Cleanroom)
- Chromosomal alleles govern movement speed, jump height, and material yields, supporting Mendelian inheritance, crossover, and flux-induced mutations.

---

## 8. The Self-Feeding Ouroboros Loop & Agent Smith Protocol

```
 [ Ingested Signal / Raw Media ]
               │
               ▼  (Topological Distillation: Chebyshev Modes, Betti Numbers, CRT Residues)
 ┌───────────────────────────────────────────────────────────┐
 │       Unified Gyroidic Manifold & Resonance Cavity        │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼  (Direct Physical Translation)
 ┌───────────────────────────────────────────────────────────┐
 │                 Voxelboxter Simulation                    │
 │  • KAN B-Spline Surfaces (BSplineCompiledMod)             │
 │  • Mohr-Coulomb / Drucker-Prager Dual-Yield Plasticity    │
 │  • FBM Anisotropic Erosion (Tire Marks, Ley Lines)        │
 │  • Morton Octree Streaming & Pirangi Cashew Flora         │
 │  • ISN Fauna Dynamics & Defect Propagation PDE            │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼  (World Collision / Rupture Scars)
 ┌───────────────────────────────────────────────────────────┐
 │     Ouroboros Shadow Logging & Dyad Fossilization         │
 │  (Fossilizes world scars into Poincaré Gravity Wells)     │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼  (P2P Agent Smith Mesh Sharing)
 ┌───────────────────────────────────────────────────────────┐
 │        Decoupled Syntax Synced Across Peer Nodes          │
 └───────────────────────────────────────────────────────────┘
```

When a physical event occurs in Voxelboxter—a crash, a track fracture, a carved gully, or a handcrafted tool—the event is not lost:
1. **Ouroboros Shadow Logging**:
   `DiegeticEngine` captures anomalies and ruptures (`[SHADOW LOG]`), packaging them into Knowledge Dyads.
2. **Dyad Fossilization**:
   The `DyadFossilizer` locks these dyads into persistent `.pt` artifacts anchored to the Poincaré disk. These scars become **Confabulation Gravity Wells** guiding future reasoner passes.
3. **The Agent Smith Protocol**:
   When a dyad achieves topological stability ($PAS_h \ge \theta_L$), its irreducible syntax (CRT residues, prime ladders, Betti numbers) is exported via `fossilizer.export_agent_smith()` into a portable payload decoupled from local hardware friction ($t_{\text{RFC}}$ DRAM stalls). Peer instances on the Bonfire / Freenet mesh download the payload and inject its B-spline skeleton directly into their local lattice gaps (`requires_grad=False`), gaining structural capabilities without retraining passes.

---

## 7. Cleanroom Mechanics Suite & Ecological Manifestation

The cleanroom mechanics suite implemented in [cleanroom_mechanics.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py) provides non-dual, mathematically grounded implementations of industrial and ecological modded gameplay systems:

### 7.1 JourneyMap Topological Radar (`JourneyTopoRadar`)
Real-time subterranean entity scanning and beacon distance calculation. Deconstructs Minecraft JourneyMap radar mechanics into coordinate distance matrices and persistent `TopoWaypoint` death fossilation markers.

### 7.2 Just Enough Resources Depth Curves (`ResourceDistributionInspector`)
Replaces ad-hoc block spawn tables with true orthogonal Chebyshev polynomial curves evaluated along world depth intervals $y \in [-64, 320]$, matching [MinimaxPolynomialApproximation](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/tda/chebyshev_filtration.py#L30) equioscillation principles.

### 7.3 Multi-Mine Progressive Fracture (`MultiMineMemory`)
Cleanroom implementation of AtomicStryker's Multi Mine. Remembers partially fractured voxel blocks across multiple agents, tool hits, and time ticks. When accumulated damage reaches 1.0, the block fractures from [PointerlessOctree](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L767) and deposits harvested scrap mass into [InventoryComponent](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L52).

### 7.4 Mekanism Multi-Tier Metallurgical Dissolution (`MekanismProcessingPipeline`)
Models progressive 1x to 5x ore refinement:
* **Tier 1 (Smelting):** 1x yield (Ore -> Ingot).
* **Tier 2 (Enrichment):** 2x yield via mechanical pulverization.
* **Tier 3 (Purification):** 3x yield utilizing oxygen gas bubbling.
* **Tier 4 (Chemical Injection):** 4x yield utilizing gaseous hydrogen chloride (HCl).
* **Tier 5 (Dissolution & Crystallization):** 5x yield utilizing sulfuric acid ($H_2SO_4$) slurry synthesis.
Monitored by [CarnotMobiusLedger](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/carnot_mobius_ledger.py#L25) to account for thermodynamic waste heat and entropy production.

### 7.5 SEM TECH: Saltwater Electro Mining (`SEMElectrochemicalExtractor`)
* **Inspiration:** SEM TECH / Rowow LLC (Robert Karas, open-source CERN-OHL-S v2).
* **Principles:** Closed-loop hydrometallurgical leaching utilizing ambient saltwater brine ($NaCl + H_2O$) and low-voltage DC / RF electricity. Generates in-situ nascent chlorine and hypochlorite oxidizers to leach and cathode-electrodeposit gold (Au), silver (Ag), platinum group metals (PGMs), copper (Cu), nickel (Ni), and rare earth elements directly from accumulated low-grade waste tailings, slag, and gangue stockpiles.
* **Closed-Loop Recirculation:** Recovers >95% of saline electrolyte in a closed cycle, allowing early-game players to extract high-value metals without late-game sulfuric acid plants.

### 7.6 Create: Aeronautics Rigid Assemblies (`AeronauticContraption`)
Eliminates duplicated kinematics by directly composing canonical [RigidBody](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L110), [Propulsor](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L140), [VehicleEngine](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L190), [AirBreathingBattery](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L170), and [EnvironmentalAtmosphere](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/voxelboxter_simulation.py#L210). Computes Archimedes buoyant envelope displacement ($F_{\text{buoyant}} = (\rho_{\text{air}} - \rho_{\text{gas}}) V g$) and sleeve-valve combustion propeller thrust, constrained by [DruckerPragerProjection](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/yield_criteria.py#L42) structural frame shear failure.

### 7.7 Farmer's Delight & Spice of Life Nutrition (`NutritionalDiversityTracker`)
Combines Carrot Edition (permanent milestone max HP increases as the player discovers unique foods) with Onion Edition (rolling Shannon entropy variety scoring, conferring positive buffs for diverse diets and sluggish malaise for monotonous diets).

### 7.8 Single-Voxel Conduits & 16-Color Bundled Cabling (`CompositeVoxelConduit` & `RedNet16BundledCable`)
Enables power (FE), fluids (mB), items, and redstone signals to route through a single voxel space without collision. RedNet bundled cables isolate 16 independent analog subnets (0-255) through a single connection.

### 7.9 Dual-Lane Logistics & Directional Inserters (`TransportBeltSegment` & `DirectionalInserter`)
Factorio-style item transport featuring two independent belt lanes (15 to 45 items/s throughput) and directional mechanical inserters with swing speed, filter selection, and stack capacities.

### 7.10 Mendelian Animal Husbandry (`FaunaGeneticsComponent`)
Phenotypic inheritance with chromosomal alleles governing movement speed, jump height, and material yields, subject to crossover and ambient mutation rates.

---

## 8. Node Environment Execution Hooks & Dual-Yield Interaction

All mechanics in and outside of [cleanroom_mechanics.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/environment/cleanroom_mechanics.py) are directly wired into [node_environment.py](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/scripting/node_environment.py) via dedicated hook methods in `PhysicalNodeEditor`:

1. `hook_cleanroom_radar()`: JourneyMap subterranean radar scanning and death waypoints.
2. `hook_cleanroom_jer()`: JER Chebyshev ore distributions and mob drops.
3. `hook_cleanroom_multi_mine()`: AtomicStryker multi-block fracture and mass recovery.
4. `hook_cleanroom_mekanism()`: Mekanism 1x-5x metallurgical refining with Carnot thermodynamic telemetry.
5. `hook_cleanroom_sem_tech()`: SEM Tech saltwater + electricity tailings extraction and brine recycling.
6. `hook_cleanroom_aeronautics()`: Create: Aeronautics airship lift, thrust, and Drucker-Prager structural stress.
7. `hook_cleanroom_nutrition()`: Spice of Life Carrot/Onion HP milestones and Shannon entropy.
8. `hook_cleanroom_conduits_rednet()`: EnderIO multi-bus conduits and RedNet 16-color channels.
9. `hook_cleanroom_logistics()`: Factorio transport belts and directional inserters.
10. `hook_cleanroom_mob()`: AtomicStryker Infernal Mobs affixes and AdaptiveSkeletonHarness morphology.
11. `hook_cleanroom_fauna_genetics()`: Animal Husbandry Mendelian genetics and offspring breeding.
12. `hook_cleanroom_inventory()`: ExpandedInventorySystem item/fluid/gas stack management.
13. `hook_chisel_octree()`: Chisels & Bits Morton-encoded micro-voxel carving and placement.
14. `hook_adaptive_rig()`: Procedural IK skeleton generation via [AdaptiveSkeletonHarness](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/adaptive_skeleton_harness.py#L20).
15. `hook_dual_yield_stress()`: Geotechnical dual-regime plasticity via [MohrCoulombProjection](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/yield_criteria.py#L13) (sharp local shear) and [DruckerPragerProjection](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/core/yield_criteria.py#L42) (smooth global adaptation envelope).

These hooks are continuously evaluated inside `evaluate_tick()` and asynchronously animated in DearPyGui frames through `_evaluation_worker()`, outputting live telemetry strings directly to node graph sockets.
