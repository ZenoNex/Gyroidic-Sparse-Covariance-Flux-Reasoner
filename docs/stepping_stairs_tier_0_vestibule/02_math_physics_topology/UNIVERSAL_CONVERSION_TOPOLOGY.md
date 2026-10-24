# Universal Conversion Topology

## Overview
Inspired by the universal converter (`p2r3/convert`), the Gyroidic Architecture implements a **Universal Topology Converter** (`src/data/universal_topology_converter.py`). However, unlike standard converters which perform format translation on literal data (thereby risking copyright infringement by ingesting or generating verbatim text, meshes, or media), our converter extracts **Structural Causality and Topological Proxies**.

## The Philosophy of Nonobstructive Embeddings

As laid out in `THE_VOYNICH_ARCHITECTURE.md`, our AI must focus on the physics of the underlying data rather than its surface-level human interpretation. 
If we ingest a 3D `.obj` file representing a copyrighted character, extracting the exact triangles (Euclidean mesh) is a copyright risk. Extracting the **Betti numbers** (how many holes the character has), the **vertex density proxy**, and the **surface-area-to-volume ratio** is fundamentally a mathematical observation of its topological space. 

This gives our Gyroidic Reasoner enough spatial context to calculate Mohr-Coulomb yield stresses and resonance potentials without ever possessing the copyrighted object.

## The Deep Tensor Information Theory Approach

Previously, the system attempted to use "straw narrow" scalar proxies (e.g., extracting purely Betti numbers from 3D models or paragraph counts from documents). This approach was rejected because it failed to provide the high-dimensional bandwidth required for the Gyroidic engine to perform non-obstructive reasoning. 

Instead, the `UniversalTopologyConverter` now relies on structural information theory, generating the following tensor fields from the raw byte entropy of **any** file format:

### 1. Spectral Tensor Projection
We compute a normalized 256-bin byte frequency histogram and project it through the `PolynomialBasis` (Chebyshev polynomials) into the `[1, 768]` high-dimensional phase space. This evaluates the exact statistical "shape" of the file without retaining its semantic content.

### 2. Polynomial CRT Pressure Signature
To avoid the "hard-coded prime heresy," the converter dynamically fetches resonant primes via `get_prime_ladder(num_moduli)`. It uses these co-prime moduli to construct a Chinese Remainder Theorem (CRT) pressure signature, representing the file's resistance to modular collapse.

### 3. Defect Scout Anomalies
By calculating the Z-score of the byte distribution, we identify extremely sparse, negative-entropy regions (where frequency drops significantly below the mean). These anomalies are passed as a `[1, 256]` tensor representing structural defects, analogous to geometric yield points.

### 4. Media Delegation
When native structural reasoning requires intrinsic temporal analysis (e.g., video or audio), the converter delegates directly to the `IVSTEncoder`, avoiding reimplementation of the intrinsic volume and spectral tensor extractions.

## The Output Boundary Policy
All extractions enforce the `OUTPUT_BOUNDARY_POLICY.md` strictures:
1.  **Finite-Only Outputs**: No NaNs or Infs leak into the structural embedding.
2.  **Universal Signature**: All conversions are signed with an `honesty_jitter` signature, verifying that the conversion resulted from the system's own topology probes rather than an external injection.
