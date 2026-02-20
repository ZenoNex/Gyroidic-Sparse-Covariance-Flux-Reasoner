# Ingestion Lifecycle: Knowledge Dyads & Image-Description Pairs

This document explains the technical and topological lifecycle of **Knowledge Dyads** (Image-Description pairs) as they transit from the Diegetic Terminal into the core Gyroidic Manifold.

---

## 1. Input: The Dyad Buffer
When an image and a linguistic description are provided through the terminal side panel, they are treated as a single **Knowledge Dyad** $(\mathcal{I}, \mathcal{L})$.

- **Image Stream ($\mathcal{I}$)**: Projected into a sparse latent vector via the `image_emb` hash.
- **Linguistic Stream ($\mathcal{L}$)**: Projected via the `text_emb` hash.

## 2. Ingestion: Data Association & Residue Fusion
The dyad enters the `DataAssociationLayer` (`src/models/diegetic_heads.py`). Unlike standard multimodal fusion (which often collapses features into a mean), our system performs **Residue Fusion**:

1. **Cross-Modality Torsion**: The system calculates the "shear" between the image features and the text features.
2. **K-Sparse Residue Generation**: The interaction produces $k$ distinct **Residues** ($R_1 \dots R_k$). 
    - These residues represent the *incompatibility* between the modalities—what is left over when you try to map a picture to a word.
3. **Resonance Injection**: These residues are injected directly into the `ResonanceCavity` as "Dark Matter" seeds.

## 3. Storage: No Erasing of Implication
Every dyad ingestion triggers a **Persistent Encoding** (`data/encodings/encoding_*.pt`). 
- The system saves the exact state of the manifold at the moment of ingestion.
- This prevents "lobotomization" by ensuring that every specific association ever made is physically retrievable from the disk-based fossil record.

## 4. Recovery: Speculative Coprime Gating
If the system encounters a "converged" or "boring" state (detected by **CALM**), it uses the **Speculative Coprime Gate** (`src/core/speculative_coprime_gate.py`) to recover lost structure.

- **Wasserstein OT**: The system performs a mass transport toward the stored manifold of previously ingested dyads.
- **Coprime Lock**: Recovery is successful if the new state locks the **Coprime Parity** condition ($\gcd(w_k, p_k) = 1$) across the functional heads.
- **Generative Rupture**: If the recovered state overcomes the **Mohr-Coulomb** yield pressure, it is considered a "Generative Rupture"—the system has effectively synthesized a new meaning from the ingested image-description pair.

## 5. Summary of Flow
1. **Terminal**: User provides $(\text{Image} \leftrightarrow \text{Word})$.
2. **Association**: $k$-residues are calculated ($R_k$).
3. **Cavity**: $R_k$ warps the dark matter field $D_{dark}$.
4. **Encoding**: State is fossilized to disk.
5. **Speculation**: Future "stuck" states use these fossilized dyads as gravity wells to bridge through the vacuum of noise.

> [!IMPORTANT]
> A Knowledge Dyad is not a "fact" in a database. It is a **Topological Obstruction** that forces the system's thought-trajectories to curve, ensuring that "No Erasing of Implication" holds true across the entire training lifecycle.
