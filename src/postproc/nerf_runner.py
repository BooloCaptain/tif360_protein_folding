import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import mp_nerf.massive_pnerf as mp_massive  # type: ignore
except Exception:
    mp_massive = None

# Sequential NeRF (Natural Extension of Reference Frame) reconstruction for C-alpha
# Given per-residue internal coordinates: distance d (to previous atom), bond angle theta,
# and dihedral tau, reconstruct Cartesian coordinates for a chain of points.


def _build_initial_triangle(d1=1.0, d2=1.0, angle=np.pi/2.0):
    # place first atom at origin, second on x axis, third in xy-plane
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([d1, 0.0, 0.0])
    # third point using bond length d2 and angle
    x = d2 * np.cos(angle)
    y = d2 * np.sin(angle)
    p2 = np.array([d1 - x, y, 0.0])
    return p0, p1, p2


def next_coord(p_im3, p_im2, p_im1, bond_length, bond_angle, dihedral):
    # Vector from i-2 to i-1
    v1 = p_im1 - p_im2
    v2 = p_im2 - p_im3
    # Normalize
    e1 = v1 / np.linalg.norm(v1)
    e2 = v2 / np.linalg.norm(v2)
    # Build orthonormal basis
    n = np.cross(e2, e1)
    n = n / (np.linalg.norm(n) + 1e-12)
    b = np.cross(e1, n)
    # position in local coordinates
    # Using standard internal coordinate conversion
    x_local = -bond_length * np.cos(bond_angle)
    y_local = bond_length * np.sin(bond_angle) * np.cos(dihedral)
    z_local = bond_length * np.sin(bond_angle) * np.sin(dihedral)
    # Map to global
    return p_im1 + x_local * e1 + y_local * b + z_local * n


def reconstruct_chain(internals):
    """Reconstruct chain coordinates from internals.

    internals: array-like shape (L,3) where each row is [d, theta, tau]
    Returns coords array shape (L,3)
    """
    internals = np.asarray(internals)
    L = internals.shape[0]
    if L == 0:
        return np.zeros((0,3))
    # initialize first three points using defaults from first two internals
    d1 = internals[0,0] if L>0 else 1.0
    d2 = internals[1,0] if L>1 else d1
    theta1 = internals[0,1] if L>0 else np.pi/2.0
    p0, p1, p2 = _build_initial_triangle(d1, d2, theta1)
    coords = [p0, p1, p2]
    for i in range(3, L):
        d, theta, tau = internals[i,0], internals[i,1], internals[i,2]
        p = next_coord(coords[i-3], coords[i-2], coords[i-1], d, theta, tau)
        coords.append(p)
    coords = np.stack(coords[:L], axis=0)
    return coords


def batch_reconstruct(batch_internals):
    """Reconstruct for a batch: batch_internals is list of (L_i,3) arrays."""
    out = []
    for internals in batch_internals:
        out.append(reconstruct_chain(internals))
    return out


def batch_reconstruct_parallel(batch_internals):
    """Use MP-NeRF when available, otherwise fallback to sequential.

    Uses batched mp_nerf_torch calls across active chains per residue index.
    """
    if mp_massive is None or torch is None:
        return batch_reconstruct(batch_internals)

    try:
        # Prepare padded batched internals.
        lengths = [arr.shape[0] for arr in batch_internals]
        if not lengths:
            return []
        max_len = max(lengths)
        B = len(batch_internals)
        d = np.zeros((B, max_len), dtype=np.float32)
        theta = np.zeros((B, max_len), dtype=np.float32)
        tau = np.zeros((B, max_len), dtype=np.float32)
        for b, arr in enumerate(batch_internals):
            L = arr.shape[0]
            d[b, :L] = arr[:, 0]
            theta[b, :L] = arr[:, 1]
            tau[b, :L] = arr[:, 2]

        coords = np.zeros((B, max_len, 3), dtype=np.float32)
        for b in range(B):
            L = lengths[b]
            if L == 0:
                continue
            d1 = d[b, 0]
            d2 = d[b, 1] if L > 1 else d1
            ang = theta[b, 0]
            p0, p1, p2 = _build_initial_triangle(d1, d2, ang)
            coords[b, 0] = p0
            if L > 1:
                coords[b, 1] = p1
            if L > 2:
                coords[b, 2] = p2

        # Vectorized chain growth using mp-nerf primitive.
        for i in range(3, max_len):
            active = [b for b, L in enumerate(lengths) if L > i]
            if not active:
                continue
            a = torch.tensor(coords[active, i - 3, :], dtype=torch.float32)
            b = torch.tensor(coords[active, i - 2, :], dtype=torch.float32)
            c = torch.tensor(coords[active, i - 1, :], dtype=torch.float32)
            l = torch.tensor(d[active, i], dtype=torch.float32)
            t = torch.tensor(theta[active, i], dtype=torch.float32)
            x = torch.tensor(tau[active, i], dtype=torch.float32)
            nxt = mp_massive.mp_nerf_torch(a, b, c, l, t, x).detach().cpu().numpy()
            coords[active, i, :] = nxt

        return [coords[b, :lengths[b], :] for b in range(B)]
    except Exception:
        return batch_reconstruct(batch_internals)
