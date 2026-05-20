import torch
import torch.nn.functional as F


def safe_cdist(x, y, eps=1e-8):
    """Computes pairwise distance avoiding NaN gradients at exactly 0."""
    # [FIX 2]: Adds epsilon before sqrt to prevent division by zero in backward pass
    diff = x.unsqueeze(2) - y.unsqueeze(1) # (B, L, L, 3)
    sq_dist = (diff ** 2).sum(dim=-1)
    return torch.sqrt(sq_dist + eps)


def end_to_end_loss(pred_1d, target_angles, target_distances, pred_coords, target_coords, 
                    ss_logits=None, target_ss=None, 
                    lambda_dist=1.0, lambda_3d=1.0, lambda_ss=0.5, 
                    mask_1d=None, mask_3d=None, mask=None, band_mask_size=30):
    """
    Computes both Local (Angle/Dist), Global (3D dRMSD), and Auxiliary (SS) losses.
    """
    B, L = target_angles.shape[0], target_angles.shape[1]

    # Scrub SCN dataset NaNs BEFORE they touch any mathematical operations
    target_angles = torch.nan_to_num(target_angles, nan=0.0)
    target_distances = torch.nan_to_num(target_distances, nan=0.0)
    target_coords = torch.nan_to_num(target_coords, nan=0.0)

    # ==========================================
    # 1. LOCAL KINEMATIC LOSS (Angles + Bonds)
    # ==========================================
    pred_theta = F.normalize(pred_1d[..., 0:2], p=2, dim=-1, eps=1e-8)  
    pred_tau = F.normalize(pred_1d[..., 2:4], p=2, dim=-1, eps=1e-8)    
    pred_d = pred_1d[..., 4]        

    target_theta = torch.stack([torch.sin(target_angles[..., 0]), torch.cos(target_angles[..., 0])], dim=-1)
    target_tau = torch.stack([torch.sin(target_angles[..., 1]), torch.cos(target_angles[..., 1])], dim=-1)

    mse_theta = torch.sum(torch.abs(pred_theta - target_theta), dim=-1)
    mse_tau = torch.sum(torch.abs(pred_tau - target_tau), dim=-1)
    
    sin_theta_target = torch.abs(target_theta[..., 0])
    mse_tau = mse_tau * sin_theta_target
    
    mse_trig_unreduced = (mse_theta + mse_tau) * 0.5
    mse_dist_unreduced = F.huber_loss(pred_d, target_distances, reduction='none', delta=1.0) 

    # ==========================================
    # 2. GLOBAL STRUCTURAL LOSS (dRMSD on C-alpha)
    # ==========================================
    pred_ca = pred_coords
    target_ca = target_coords

    pred_pdists = safe_cdist(pred_ca, pred_ca)
    target_pdists = safe_cdist(target_ca, target_ca)

    drmsd_error = torch.abs(pred_pdists - target_pdists)
    drmsd_error = torch.clamp(drmsd_error, max=10.0)
    drmsd_unreduced = F.huber_loss(drmsd_error, torch.zeros_like(drmsd_error), reduction='none', delta=2.0)

    # ==========================================
    # 3. MASKING & REDUCTION
    # ==========================================
    if mask_1d is None and mask is not None:
        mask_1d = mask
    if mask_3d is None and mask is not None:
        mask_3d = mask

    if mask_1d is not None and mask_3d is not None:
        mask_1d = mask_1d.float()
        mask_3d = mask_3d.float()
        
        mask_2d = mask_3d.unsqueeze(-1) * mask_3d.unsqueeze(-2) # (B, L, L)
        
        idx = torch.arange(L, device=pred_coords.device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) # (L, L)
        
        band_mask = (seq_sep < band_mask_size).float().unsqueeze(0) # (1, L, L)
        mask_2d = mask_2d * band_mask
        
        mse_trig_masked = torch.where(mask_1d.bool(), mse_trig_unreduced, 0.0)
        mse_dist_masked = torch.where(mask_1d.bool(), mse_dist_unreduced, 0.0)
        drmsd_masked = torch.where(mask_2d.bool(), drmsd_unreduced, 0.0)
        
        valid_tokens_1d = mask_1d.sum() + 1e-8
        valid_pairs_2d = mask_2d.sum() + 1e-8
        
        mse_trig = mse_trig_masked.sum() / valid_tokens_1d
        mse_dist = mse_dist_masked.sum() / valid_tokens_1d
        loss_3d = drmsd_masked.sum() / valid_pairs_2d
    else:
        mse_trig = mse_trig_unreduced.mean()
        mse_dist = mse_dist_unreduced.mean()
        loss_3d = drmsd_unreduced.mean()

    loss_ss = torch.tensor(0.0, device=pred_1d.device)
    
    if ss_logits is not None and target_ss is not None:
        target_ss_masked = target_ss.clone()
        if mask_1d is not None:
            target_ss_masked[mask_1d == 0] = 3
            
        if (target_ss_masked != 3).any():
            loss_ss = F.cross_entropy(
                ss_logits.reshape(-1, 3),
                target_ss_masked.reshape(-1), 
                ignore_index=3,
                label_smoothing=0.1
            )

    # Combine Local, Global, and Auxiliary losses
    total_loss = mse_trig + (lambda_dist * mse_dist) + (lambda_3d * loss_3d) + (lambda_ss * loss_ss)
    
    return total_loss, mse_trig, mse_dist, loss_3d, loss_ss