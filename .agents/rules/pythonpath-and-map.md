# PYTHONPATH Rule
When running Python scripts in this repository, always ensure that `$env:PYTHONPATH="."` is set in PowerShell so that the `src` module can be resolved.
Example: `$env:PYTHONPATH="."; .venv\scripts\python.exe tests\test_unicorn_synthesis.py`

# Excluded Root Hazard Directories
Never run unfiltered recursive searches across the workspace root. Exclude `.venv`, `data`, `datasets`, `node_modules`, `wikipedia extracter`, `gyroidic reasoner related media`, `test_raw_data`, `test_storage`, `test_repo`, `recovered_files`, `tmp`, `scratch`, `google secret`, `.git`, `__pycache__`, `.pytest_cache`, `.qodo`, and `.zk_proofs`.

# Rule-Bearing Documentation & Canonical Maps in `docs/`
The following files in `docs/` function directly as architectural rules, mathematical invariants, and structural maps:
- `docs/STEPPING_STAIRS_NAVIGATIONAL_TREE.md`: Master stepping stairs navigation map.
- `docs/UNIVERSAL_ORCHESTRATOR_REHYBRIDIZATION.md`: Master blueprint mapping subsystems into `hybrid_backend.py` and `src/core/orchestrator.py`.
- `docs/stepping_stairs_tier_0_vestibule/01_architecture_core/MODULE_INVENTORY.md`: Complete inventory of modules and classes across `src/`.
- `docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/MATHEMATICAL_DETAILS.md`: Governing equations, PAS_h, APAS_zeta, Prime/Fibonacci/Catalan invariants, and FGRT operators.
- `docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/TOPOLOGICAL_EXTENSIONS.md`: Topological extensions, Stasheff Associahedron bracketings, and manifold triangulations.
- `docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/MATRYOSHKA_POLYTOPE_MAP.md`: Nested Matrioshka polytope containment and Catalan shell parenthesizations.
- `docs/stepping_stairs_tier_0_vestibule/02_math_physics_topology/KINEMATIC_RESONANCE_MAPPINGS.md`: Kinematic resonance and structural mappings.
- `docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/03_philosophy_lore/NOMENCLATURE.md`: Canonical terminology and anti-lobotomy definitions.
- `docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/04_components_and_mechanics/PRIME_RESONANCE_LADDER.md`: Prime Resonance Ladder, Repunit mirrors, Fibonacci-Prime entropy, and Catalan Associahedron non-associative operator bracketings.
- `docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/stepping_stairs_tier_2_crypt/05_implementation_and_guides/CODES_Implementation_Mapping.md`: CODES v40 invariant-to-code mapping.
- `docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/stepping_stairs_tier_2_crypt/05_implementation_and_guides/IMPLEMENTATION_PLAN_FORMALISM.md`: Implementation plan formalism.
- `docs/stepping_stairs_tier_0_vestibule/stepping_stairs_tier_1_mezzanine/stepping_stairs_tier_2_crypt/05_implementation_and_guides/OPEN_QUESTIONS.md`: Open mathematical questions and architectural resolutions.

# Project Map
- `docs/`: Theoretical documentation, architectural specifications, and stepping stairs landings.
  - `STEPPING_STAIRS_NAVIGATIONAL_TREE.md`: Master stepping stairs navigation map.
  - `UNIVERSAL_ORCHESTRATOR_REHYBRIDIZATION.md`: Master blueprint mapping isolated systems into `hybrid_backend.py` and `src/core/orchestrator.py`.
  - `stepping_stairs_tier_0_vestibule/`: Tier 0 Vestibule landing.
    - `00_entrance_to_lower_level/`: Subterranean gateway portal.
    - `01_architecture_core/`: Core architecture, system constitution, and module inventory.
    - `02_math_physics_topology/`: Mathematical details (FGRT, Atiyah-Singer index, manifolds, ADMR).
    - `stepping_stairs_tier_1_mezzanine/`: Tier 1 Mezzanine landing.
      - `03_philosophy_lore/`: Non-dual dynamics, archetypes, and TADC lore.
      - `04_components_and_mechanics/`: Scripting systems, garden attractors, and prime resonance ladders.
      - `stepping_stairs_tier_2_crypt/`: Tier 2 Crypt landing.
        - `05_implementation_and_guides/`: Command guides, Codes v40, and implementation integrity.
        - `06_reports_and_inspirations/`: Breakthrough reports, visual artifacts, and inspirations.
        - `extended coherence research/`: RP4 cosmology and BAO zero-parameter validations.
        - `summaries/`: Historical audit logs and component summaries.
- `src/`: Implementation of the Gyroidic Sparse Covariance Flux Reasoner.
  - `hybrid_backend.py`: The unified runtime runner and HTTP/WebSocket server.
  - `core/`: The reasoning engine core (orchestrator, invariants, FGRT primitives, ADMR, evidence modules, energy monitor).
  - `topology/`: Advanced geometry systems (PyOpenCL gyroid covariance, hyper-rings, Moebius bundles).
  - `scripting/`: Physical scripting node graph (`node_environment.py`).
  - `models/`: Neural and symbolic reasoning heads (gyroid reasoner, polynomial embeddings, resonance cavity).
  - `surrogates/`: CALM and KAGH fast meta-learning surrogates.
  - `safety/`: Red teaming, trust inheritance, and subversive oracle probes.
  - `training/`: Enhanced temporal training, FGRT structural trainer, and EBM adaptors.
  - `ui/`: Diegetic backend, Voxelboxter simulation, and conversational GUI interfaces.
  - `data/`: Sovereign knowledge ingestion (ArXiv, bulletin boards, Wikipedia, financial feeds).
  - `codec/`: Gyroidic codec, conformal log-polar projection, `image_extension.py`, and vision surgery.
  - `p2p/`: Bonfire nomadic ring consensus (`ShamirThresholdSharing`, `NonEulerShamirTopologySharing`) and Freenet client.
- `tests/`: Tiered unit, integration, and stress test suites (`01_architecture_core/` .. `05_ingestion_network_and_infra/`).
- `.agents/`: Operational rules and integrity guidelines.

# Living Interaction Map
The detailed Mermaid interaction diagrams and Growth Log live in [`.agents/rules/living_interaction_map.md`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/.agents/rules/living_interaction_map.md) (split out to keep rule files modular and concise).
