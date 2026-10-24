import math
import torch
import torch.nn.functional as F

def get_russian_doll_projection(coeffs: torch.Tensor, k_image_max: int = 32) -> torch.Tensor:
    """
    Extracts nested 'Russian Doll' Chebyshev projections from raw coefficients.
    
    Args:
        coeffs: Raw coefficients [3 * k_image_max] (L, Cr, Cb)
        k_image_max: Max degree per channel (default 32)
        
    Returns:
        composite_residue: Normalized average of Degrees 8, 16, and 32 projections.
    """
    target = k_image_max * 3
    if coeffs.numel() < target:
        coeffs = F.pad(coeffs, (0, target - coeffs.numel()))
    else:
        coeffs = coeffs[:target]
        
    # Degree 8 projection (Outer Shell)
    deg8_mask = torch.zeros_like(coeffs)
    deg8_indices = [i for i in range(target) if (i % k_image_max) < 8]
    deg8_mask[deg8_indices] = 1.0
    p8 = (coeffs * deg8_mask)
    
    # Degree 16 projection (Middle Shell)
    deg16_mask = torch.zeros_like(coeffs)
    deg16_indices = [i for i in range(target) if (i % k_image_max) < 16]
    deg16_mask[deg16_indices] = 1.0
    p16 = (coeffs * deg16_mask)
    
    # Degree 32 projection (Inner Shell) - uses full coefficients
    p32 = coeffs
    
    # Composite Russian Doll residue (raw coefficient space)
    # This represents the interlaced structural signal
    return (p8 + p16 + p32) / 3.0


def project_chromata_filaments(trajectories: list, target_dim: int = 96) -> torch.Tensor:
    """
    Projects Chromata streamline filaments into a compact topological embedding vector.
    Extracts arc-length distribution and winding invariants.
    """
    lengths = []
    windings = []
    for path in trajectories:
        if len(path) < 2:
            continue
        l = 0.0
        angle = 0.0
        prev_a = None
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            seg = (dx * dx + dy * dy)**0.5
            l += seg
            if seg > 1e-4:
                curr_a = math.atan2(dy, dx) if 'math' in globals() else 0.0
                if prev_a is not None:
                    d = curr_a - prev_a
                    angle += abs(d)
                prev_a = curr_a
        lengths.append(l)
        windings.append(angle / (2.0 * 3.14159265))

    vec = torch.zeros(target_dim, dtype=torch.float32)
    if lengths:
        t_l = torch.tensor(lengths, dtype=torch.float32)
        l_norm = t_l / (t_l.sum() + 1e-8)
        n = min(len(l_norm), target_dim // 2)
        vec[:n] = l_norm[:n]
    if windings:
        t_w = torch.tensor(windings, dtype=torch.float32)
        w_norm = t_w / (t_w.sum() + 1e-8)
        n = min(len(w_norm), target_dim // 2)
        vec[target_dim // 2 : target_dim // 2 + n] = w_norm[:n]

    return vec


def project_geometrize_primitives(primitives: list, target_dim: int = 96) -> torch.Tensor:
    """
    Projects Geometrize continuous parametric primitives into a compact mass-moment tensor vector.
    Bypasses discrete pixel and voxel bloat.
    """
    vec = torch.zeros(target_dim, dtype=torch.float32)
    idx = 0
    for prim in primitives:
        if idx + 7 > target_dim:
            break
        c = prim.get("center", (0.0, 0.0, 0.0))
        r = prim.get("radii", (1.0, 1.0, 1.0))
        op = prim.get("opacity", 1.0)
        vec[idx] = float(c[0])
        vec[idx+1] = float(c[1])
        vec[idx+2] = float(c[2])
        vec[idx+3] = float(r[0])
        vec[idx+4] = float(r[1])
        vec[idx+5] = float(r[2])
        vec[idx+6] = float(op)
        idx += 7

    norm = torch.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def project_multimodal_fingerprint_to_96(payload) -> torch.Tensor:
    """
    Unified projection turning multimodal fingerprint payloads (Chromata streamlines,
    Geometrize primitives, multi-basis spectral dictionaries, or raw tensors) into
    a canonical 96-dim tensor suitable for GyroidicCodec and KnowledgeDyadFossilizer.
    """
    if isinstance(payload, torch.Tensor):
        flat = payload.flatten()
        if flat.numel() < 96:
            return F.pad(flat, (0, 96 - flat.numel()))
        return flat[:96]
        
    if isinstance(payload, list):
        if len(payload) > 0 and isinstance(payload[0], list):
            # Trajectories (Chromata)
            return project_chromata_filaments(payload, target_dim=96)
        elif len(payload) > 0 and isinstance(payload[0], dict):
            # Primitives (Geometrize)
            return project_geometrize_primitives(payload, target_dim=96)
        else:
            t = torch.tensor(payload, dtype=torch.float32)
            if t.numel() < 96:
                return F.pad(t, (0, 96 - t.numel()))
            return t[:96]

    if isinstance(payload, dict):
        if "trajectories" in payload or "streamlines" in payload:
            trajs = payload.get("trajectories") or payload.get("streamlines", [])
            return project_chromata_filaments(trajs, target_dim=96)
        elif "primitives" in payload:
            return project_geometrize_primitives(payload.get("primitives", []), target_dim=96)
        elif "L" in payload and "Cr" in payload and "Cb" in payload:
            l_coeffs = payload.get("L", [])
            cr_coeffs = payload.get("Cr", [])
            cb_coeffs = payload.get("Cb", [])
            k = max(len(l_coeffs), len(cr_coeffs), len(cb_coeffs), 32)
            flat = list(l_coeffs) + list(cr_coeffs) + list(cb_coeffs)
            t = torch.tensor(flat, dtype=torch.float32)
            return get_russian_doll_projection(t, k_image_max=k)
        elif "chebyshev" in payload:
            coeffs = payload.get("chebyshev", [])
            t = torch.tensor(coeffs, dtype=torch.float32)
            if t.numel() < 96:
                return F.pad(t, (0, 96 - t.numel()))
            return t[:96]
        elif "spectral_signature" in payload and isinstance(payload["spectral_signature"], torch.Tensor):
            return project_multimodal_fingerprint_to_96(payload["spectral_signature"])

    return torch.zeros(96, dtype=torch.float32)

