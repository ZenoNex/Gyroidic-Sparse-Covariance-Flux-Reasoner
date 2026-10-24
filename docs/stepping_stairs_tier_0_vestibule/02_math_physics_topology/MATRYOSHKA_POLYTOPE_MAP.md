# Matrioshka Nested Polytope Dynamics Map

This diagram illustrates the flow of the Gyroidic Sparse Covariance Flux Reasoner when operating within the Matrioshka nested polytope framework, as well as the transition policies between different meaning systems governed by the Chinese Remainder Theorem (CRT) index.

```mermaid
flowchart TD
    %% Define Styles
    classDef space fill:#1e1e2e,stroke:#f38ba8,stroke-width:2px,color:#cdd6f4
    classDef polytope fill:#313244,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4
    classDef process fill:#45475a,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4
    classDef error fill:#f38ba8,stroke:#11111b,color:#11111b,stroke-width:2px,font-weight:bold

    %% P_space and State Initialization
    State["Initial State (x, , l)"]
    Space["Meta-Polytope Space P_space(Context)"]

    State --> Space
    Space --> IterLevels

    subgraph IterLevels ["Matrioshka Layer Escalation (l_max down to 0)"]
        direction TB
        
        GetP["Select P^(l)_"]
        CheckContains{"Does P^(l) contain x?"}
        
        GetP --> CheckContains
        
        subgraph InsidePolytope ["Intra-Polytope Operation"]
            direction TB
            Q1["Quantize: xq = Q^(l)(x)"]
            F["Evolve: y = F(xq, l)"]
            Q2["Quantize: yq = Q^(l)(y)"]
            
            Q1 --> F --> Q2
        end
        
        CheckContains -- Yes --> InsidePolytope
        
        CheckFixedPoint{"Is yq an interior fixed point?"}
        InsidePolytope --> CheckFixedPoint
        
        StableUpdate["Stable Update: Return (yq, P^(l))"]
        CheckFixedPoint -- Yes --> StableUpdate
        
        CheckFacet{"Is yq on Facet P?"}
        CheckFixedPoint -- No --> CheckFacet
        
        FacetTransition["Facet Grazing / Switch: Return (yq, P_adjacent(yq))"]
        CheckFacet -- Yes --> FacetTransition
        
        PopOutward["Pop Outward (l = l - 1)"]
        CheckFacet -- No --> PopOutward
        PopOutward --> GetP
    end

    CheckContains -- No --> PopOutward

    NaN["NaN / BoundaryState (Topological Refusal)"]
    PopOutward -. "If l < 0" .-> NaN

    %% Apply Styles
    class State,Space space
    class InsidePolytope polytope
    class Q1,F,Q2,GetP,PopOutward process
    class StableUpdate,FacetTransition process
    class NaN error

```

## Key Architectural Principles

1. **Intra-polytope traversal (Interior):** Scalarization and traditional logic apply. The state is quantized to the layer's local resolution `(l)`.
2. **Facet Grazing (Boundary):** The state has reached an incompatibility boundary. A switch to an adjacent meaning system (via CRT index ``) is initiated.
3. **Topological Refusal (NaN / BoundaryState):** If no layer's polytope can claim the state, the system correctly refuses to map the state, throwing a `BoundaryState` stress tensor (NaN). This represents an epistemic limit, preventing "lobotomized" hallucinations outside of valid reasoning polytopes.
4. **IVST Side-Chaining and Lazarus Rehydration (Dropout):** When the state pressure drops below the chaotic envelope $y = \cos(\tau / Z)(\sin(30x) + 1)$, where $Z = 2^{-\text{level}}$ is the recursive Matrioshka shell scale, the corresponding dimensions are dropped out (`NaN`). To conserve mass, vanished energy is redistributed across the remaining active dimensions (Birkhoff constraint). The `NaN` values are then rehydrated using honest jitter via `apply_energy_based_stabilization`, breaking state collapse stasis.
