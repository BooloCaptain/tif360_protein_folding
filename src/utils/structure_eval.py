import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def build_ca_coords_nerf(bond_lengths, thetas, phis):
    """Build C-alpha coordinates with a memory-safe prefix-scan NeRF-style transform chain."""
    bond_lengths = bond_lengths.float()
    thetas = thetas.float()
    phis = phis.float()

    B, L = bond_lengths.shape
    device = bond_lengths.device
    dtype = bond_lengths.dtype

    l = bond_lengths.flatten()
    c_theta = torch.cos(thetas.flatten())
    s_theta = torch.sin(thetas.flatten())
    c_phi = torch.cos(phis.flatten())
    s_phi = torch.sin(phis.flatten())

    tmats = torch.zeros((B * L, 4, 4), device=device, dtype=dtype)

    tmats[:, 0, 0] = -c_theta
    tmats[:, 0, 1] = -s_theta
    tmats[:, 0, 3] = -l * c_theta
    tmats[:, 1, 3] = l * s_theta * c_phi
    tmats[:, 2, 3] = l * s_theta * s_phi

    tmats[:, 1, 0] = s_theta * c_phi
    tmats[:, 1, 1] = -c_theta * c_phi
    tmats[:, 1, 2] = -s_phi

    tmats[:, 2, 0] = s_theta * s_phi
    tmats[:, 2, 1] = -c_theta * s_phi
    tmats[:, 2, 2] = c_phi

    tmats[:, 3, 3] = 1.0

    tmats = tmats.view(B, L, 4, 4)

    global_tmats = tmats
    step = 1
    while step < L:
        left = global_tmats[:, :-step]
        right = global_tmats[:, step:]
        updated = torch.matmul(left, right)
        global_tmats = torch.cat([global_tmats[:, :step], updated], dim=1)
        step *= 2

    origin = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 4, 1)
    return torch.matmul(global_tmats, origin)[..., :3, 0]


def build_3d_wrapper(bond_lengths, thetas, phis):
    return build_ca_coords_nerf(bond_lengths, thetas, phis)


def angles_to_3d_coords_memory_safe(pred_1d, sequences, device):
    bond_lengths = pred_1d[..., 4]

    theta_sin = pred_1d[..., 0]
    theta_cos = pred_1d[..., 1]
    phi_sin = pred_1d[..., 2]
    phi_cos = pred_1d[..., 3]

    thetas = torch.atan2(theta_sin, theta_cos)
    phis = torch.atan2(phi_sin, phi_cos)

    return checkpoint(build_3d_wrapper, bond_lengths, thetas, phis, use_reentrant=False)


def compute_contiguous_drmsd(pred_ca, target_ca, target_ss, valid_mask, helix_idx=0, sheet_idx=1):
    """Evaluate contiguous helix/sheet distance error on a single protein."""
    if isinstance(pred_ca, np.ndarray):
        pred_ca = torch.from_numpy(pred_ca)
    if isinstance(target_ca, np.ndarray):
        target_ca = torch.from_numpy(target_ca)
    if isinstance(target_ss, np.ndarray):
        target_ss = torch.from_numpy(target_ss)
    if isinstance(valid_mask, np.ndarray):
        valid_mask = torch.from_numpy(valid_mask)

    device = pred_ca.device
    pred_ca = pred_ca.float()
    target_ca = target_ca.to(device).float()
    target_ss = target_ss.to(device).long()
    valid_mask = valid_mask.to(device).bool()

    L = pred_ca.shape[0]
    target_ss = target_ss[:L]
    valid_mask = valid_mask[:L]

    valid_idx_full = torch.where(valid_mask)[0]
    if len(valid_idx_full) > 0:
        p_sub = pred_ca[valid_idx_full]
        t_sub = target_ca[valid_idx_full]
        full_err = torch.abs(torch.cdist(p_sub, p_sub) - torch.cdist(t_sub, t_sub)).sum().item()
        full_drmsd = full_err / (len(valid_idx_full) ** 2)
    else:
        full_drmsd = float('nan')

    def get_blocks(class_idx):
        blocks = []
        curr = []
        ts_list = target_ss.cpu().tolist()
        vm_list = valid_mask.cpu().tolist()

        for i, (val, is_valid) in enumerate(zip(ts_list, vm_list)):
            if val == class_idx and is_valid:
                curr.append(i)
            else:
                if len(curr) >= 4:
                    blocks.append(curr)
                curr = []
        if len(curr) >= 4:
            blocks.append(curr)
        return blocks

    def evaluate_blocks(blocks):
        if not blocks:
            return float('nan')

        total_error = 0.0
        total_pairs = 0
        for block in blocks:
            idx = torch.tensor(block, device=device)
            p_sub = pred_ca[idx]
            t_sub = target_ca[idx]
            p_dist = torch.cdist(p_sub, p_sub)
            t_dist = torch.cdist(t_sub, t_sub)
            error = torch.abs(p_dist - t_dist)
            total_error += error.sum().item()
            total_pairs += error.numel()

        return total_error / total_pairs if total_pairs > 0 else float('nan')

    helix_blocks = get_blocks(helix_idx)
    sheet_blocks = get_blocks(sheet_idx)

    return {
        'full_drmsd': full_drmsd,
        'intra_helix_drmsd': evaluate_blocks(helix_blocks),
        'intra_sheet_drmsd': evaluate_blocks(sheet_blocks),
        'helix_count': len(helix_blocks),
        'sheet_count': len(sheet_blocks),
    }


def calculate_gdt_ts(pred, target):
    dists = np.linalg.norm(pred - target, axis=-1)
    p1 = np.mean(dists <= 1.0)
    p2 = np.mean(dists <= 2.0)
    p4 = np.mean(dists <= 4.0)
    p8 = np.mean(dists <= 8.0)
    return (p1 + p2 + p4 + p8) / 4.0


def calculate_tm_score(pred, target):
    L = len(target)
    if L <= 15:
        return 0.0
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    d0 = max(d0, 0.5)
    dists = np.linalg.norm(pred - target, axis=-1)
    return np.mean(1.0 / (1.0 + (dists / d0) ** 2))


def calculate_top_l_half_long_contact_precision(pred_coords, target_coords, threshold=8.0, seq_sep=24):
    L = len(target_coords)
    if L < seq_sep:
        return np.nan

    diff_pred = pred_coords[:, None, :] - pred_coords[None, :, :]
    dist_pred = np.linalg.norm(diff_pred, axis=-1)

    diff_tgt = target_coords[:, None, :] - target_coords[None, :, :]
    dist_tgt = np.linalg.norm(diff_tgt, axis=-1)

    i, j = np.triu_indices(L, k=seq_sep)
    if len(i) == 0:
        return np.nan

    long_pred_dists = dist_pred[i, j]
    long_tgt_dists = dist_tgt[i, j]
    top_n = max(1, L // 2)
    sort_indices = np.argsort(long_pred_dists)
    top_tgt_dists = long_tgt_dists[sort_indices[:top_n]]
    return np.sum(top_tgt_dists <= threshold) / top_n


def calculate_top_l_half_long_contact_precision_2d(contact_probs, target_coords, threshold=8.0, seq_sep=24):
    L = len(target_coords)
    if L < seq_sep:
        return np.nan

    diff_tgt = target_coords[:, None, :] - target_coords[None, :, :]
    dist_tgt = np.linalg.norm(diff_tgt, axis=-1)
    i, j = np.triu_indices(L, k=seq_sep)
    if len(i) == 0:
        return np.nan

    long_contact_probs = contact_probs[i, j]
    long_tgt_dists = dist_tgt[i, j]
    top_n = max(1, L // 2)
    sort_indices = np.argsort(long_contact_probs)[::-1]
    top_tgt_dists = long_tgt_dists[sort_indices[:top_n]]
    return np.sum(top_tgt_dists <= threshold) / top_n


def calculate_steric_clashes(pred_coords, seq_sep=3, clash_threshold=3.2):
    L = len(pred_coords)
    if L < seq_sep:
        return 0.0

    diff = pred_coords[:, None, :] - pred_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    i, j = np.triu_indices(L, k=seq_sep)
    non_adj_dists = dists[i, j]
    clashes = np.sum(non_adj_dists < clash_threshold)
    return (clashes / L) * 100
