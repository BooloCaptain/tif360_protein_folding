import torch
import torch.nn.functional as F


def safe_cdist(x, y, eps=1e-8):
    """Computes pairwise distance avoiding NaN gradients at exactly 0."""
    # [FIX 2]: Adds epsilon before sqrt to prevent division by zero in backward pass
    diff = x.unsqueeze(2) - y.unsqueeze(1) # (B, L, L, 3)
    sq_dist = (diff ** 2).sum(dim=-1)
    return torch.sqrt(sq_dist + eps)


def end_to_end_loss(pred_1d, target_angles, target_distances, pred_coords, target_coords, lambda_dist=1.0, lambda_3d=1.0, mask=None):
    """
    Computes both Local (Angle/Dist) and Global (3D dRMSD) losses.
    """
    B, L = target_angles.shape[0], target_angles.shape[1]

    # [FIX 2]: Scrub SCN dataset NaNs BEFORE they touch any mathematical operations
    target_angles = torch.nan_to_num(target_angles, nan=0.0)
    target_distances = torch.nan_to_num(target_distances, nan=0.0)
    target_coords = torch.nan_to_num(target_coords, nan=0.0)

    # ==========================================
    # 1. LOCAL KINEMATIC LOSS (Angles + Bonds)
    # ==========================================
    
    # Normalize predictions to the unit circle to prevent magnitude collapse
    pred_theta = F.normalize(pred_1d[..., 0:2], p=2, dim=-1, eps=1e-8)  
    pred_tau = F.normalize(pred_1d[..., 2:4], p=2, dim=-1, eps=1e-8)    
    pred_d = pred_1d[..., 4]        

    # Convert targets to [sin, cos]
    target_theta = torch.stack([torch.sin(target_angles[..., 0]), torch.cos(target_angles[..., 0])], dim=-1)
    target_tau = torch.stack([torch.sin(target_angles[..., 1]), torch.cos(target_angles[..., 1])], dim=-1)

    # L1 Loss for angles (prevents "parabola of apathy")
    mse_theta = torch.sum(torch.abs(pred_theta - target_theta), dim=-1)
    mse_tau = torch.sum(torch.abs(pred_tau - target_tau), dim=-1)
    
    # [THE FIX]: Weight torsion loss by the sine of the target bond angle.
    # target_theta[..., 0] is exactly torch.sin(target_angles[..., 0])
    # This zeroes out the torsion penalty when the bonds form a straight line.
    sin_theta_target = torch.abs(target_theta[..., 0])
    mse_tau = mse_tau * sin_theta_target
    
    mse_trig_unreduced = (mse_theta + mse_tau) * 0.5
    mse_dist_unreduced = F.huber_loss(pred_d, target_distances, reduction='none', delta=1.0) 

    # ==========================================
    # 2. GLOBAL STRUCTURAL LOSS (dRMSD on C-alpha)
    # ==========================================
    
    # Since we are using the custom Coarse-Grained CA-only builder,
    # pred_coords is ALREADY just the C-alpha trace! Shape: (B, L, 3)
    pred_ca = pred_coords
    
    # Your target dataset already provides JUST the Ca trace! Shape: (B, L, 3)
    target_ca = target_coords

    # Calculate all-by-all pairwise distances
    pred_pdists = safe_cdist(pred_ca, pred_ca)
    target_pdists = safe_cdist(target_ca, target_ca)

    # Calculate absolute error
    drmsd_error = torch.abs(pred_pdists - target_pdists)
    
    # Clamp the maximum error per pair to 10 Angstroms 
    # This stops the model from panicking over two atoms being 100A apart instead of 50A
    drmsd_error = torch.clamp(drmsd_error, max=10.0)

    # Apply Huber loss to the clamped error
    drmsd_unreduced = F.huber_loss(drmsd_error, torch.zeros_like(drmsd_error), reduction='none', delta=2.0)

    # ==========================================
    # 3. MASKING & REDUCTION
    # ==========================================
    
    if mask is not None:
        mask_1d = mask.float()
        
        # Create standard 2D mask for valid pairs
        mask_2d = mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(-2) # (B, L, L)
        
        # Create a matrix of index distances |i - j|
        idx = torch.arange(L, device=pred_coords.device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) # (L, L)
        
        # Only allow gradients for pairs within 30 residues of each other
        # (You can increase this to 60 or 100 later in training)
        band_mask = (seq_sep < 100).float().unsqueeze(0) # (1, L, L)
        
        # Combine the valid mask with the banding mask
        mask_2d = mask_2d * band_mask
        
        mse_trig_masked = torch.where(mask_1d.bool(), mse_trig_unreduced, 0.0)
        mse_dist_masked = torch.where(mask_1d.bool(), mse_dist_unreduced, 0.0)
        drmsd_masked = torch.where(mask_2d.bool(), drmsd_unreduced, 0.0)
        
        valid_tokens_1d = mask_1d.sum() + 1e-8
        valid_pairs_2d = mask_2d.sum() + 1e-8
        
        mse_trig = mse_trig_masked.sum() / valid_tokens_1d
        mse_dist = mse_dist_masked.sum() / valid_tokens_1d
        loss_3d = drmsd_masked.sum() / valid_pairs_2d

    # Combine Local and Global losses
    total_loss = mse_trig + (lambda_dist * mse_dist) + (lambda_3d * loss_3d)
    
    return total_loss, mse_trig, mse_dist, loss_3d