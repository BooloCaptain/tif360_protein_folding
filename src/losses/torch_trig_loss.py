import torch
import torch.nn.functional as F

def safe_cdist(x, y, eps=1e-8):
    """Computes pairwise distance avoiding NaN gradients at exactly 0."""
    diff = x.unsqueeze(2) - y.unsqueeze(1) # (B, L, L, 3)
    sq_dist = (diff ** 2).sum(dim=-1)
    return torch.sqrt(sq_dist + eps)


def end_to_end_loss(pred_1d, target_angles, target_distances, pred_coords=None, target_coords=None, 
                    ss_logits=None, target_ss=None, disto_logits=None, 
                    lambda_dist=1.0, lambda_3d=1.0, lambda_ss=0.5, lambda_disto=0.5, 
                    mask_1d=None, mask_3d=None, mask=None, band_mask_size=30):
    """
    Computes Local (Angle/Dist), Global (3D dRMSD), Auxiliary (SS), and 2D Direct Distogram losses.
    """
    B, L = target_angles.shape[0], target_angles.shape[1]
    if target_coords is None:
        raise ValueError("end_to_end_loss requires target_coords.")

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
    # [THE FIX]: Upcast coordinates to float32 before squaring/rooting to prevent variance explosions in the 3D loss
    target_pdists = safe_cdist(target_coords.float(), target_coords.float())
    if pred_coords is not None:
        pred_pdists = safe_cdist(pred_coords.float(), pred_coords.float())
        drmsd_error = torch.abs(pred_pdists - target_pdists)
        drmsd_error = torch.clamp(drmsd_error, max=10.0)
        drmsd_unreduced = F.huber_loss(drmsd_error, torch.zeros_like(drmsd_error), reduction='none', delta=2.0)
    else:
        drmsd_unreduced = None

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
        
        idx = torch.arange(L, device=pred_1d.device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) # (L, L)
        
        band_mask = (seq_sep < band_mask_size).float().unsqueeze(0) # (1, L, L)
        mask_2d = mask_2d * band_mask
        
        mse_trig_masked = torch.where(mask_1d.bool(), mse_trig_unreduced, 0.0)
        mse_dist_masked = torch.where(mask_1d.bool(), mse_dist_unreduced, 0.0)
        if drmsd_unreduced is not None:
            drmsd_masked = torch.where(mask_2d.bool(), drmsd_unreduced, 0.0)
        
        valid_tokens_1d = mask_1d.sum() + 1e-8
        valid_pairs_2d = mask_2d.sum() + 1e-8
        
        mse_trig = mse_trig_masked.sum() / valid_tokens_1d
        mse_dist = mse_dist_masked.sum() / valid_tokens_1d
        if drmsd_unreduced is not None:
            loss_3d = drmsd_masked.sum() / valid_pairs_2d
        else:
            loss_3d = torch.tensor(0.0, device=pred_1d.device)
    else:
        mse_trig = mse_trig_unreduced.mean()
        mse_dist = mse_dist_unreduced.mean()
        if drmsd_unreduced is not None:
            loss_3d = drmsd_unreduced.mean()
        else:
            loss_3d = torch.tensor(0.0, device=pred_1d.device)
        mask_2d = torch.ones_like(target_pdists)

    # ==========================================
    # 4. SECONDARY STRUCTURE AUXILIARY LOSS
    # ==========================================
    loss_ss = torch.tensor(0.0, device=pred_1d.device)
    if ss_logits is not None and target_ss is not None:
        target_ss_masked = target_ss.clone()
        if mask_1d is not None:
            target_ss_masked[mask_1d == 0] = 3
            
        if (target_ss_masked != 3).any():
            # [THE FIX]: Cast the logits to float32 before they hit the Softmax exponentiation
            loss_ss = F.cross_entropy(
                ss_logits.float().reshape(-1, 3),
                target_ss_masked.reshape(-1), 
                ignore_index=3,
                label_smoothing=0.1
            )

    # ==========================================
    # 5. DIRECT 2D DISTOGRAM LOSS (Targeted Continuous Weights)
    # ==========================================
    loss_disto = torch.tensor(0.0, device=pred_1d.device)
    if disto_logits is not None:
        num_bins = 64
        # Uniformly bin target distances between 2.0A and 22.0A
        target_bins = torch.floor((target_pdists - 2.0) / (22.0 - 2.0) * num_bins).long()
        target_bins = torch.clamp(target_bins, min=0, max=num_bins - 1)
        
        # Apply the exact same active curriculum band mask
        target_bins_masked = target_bins.clone()
        target_bins_masked[mask_2d == 0] = -100 # Standard Cross-Entropy ignore index
        
        if (target_bins_masked != -100).any():
            B, L, _ = target_bins_masked.shape
            
            # 1. Calculate Unreduced Loss [B, L, L]
            # [THE FIX]: Cast the logits to float32 before they hit the Softmax exponentiation
            raw_disto_loss = F.cross_entropy(
                disto_logits.float().permute(0, 3, 1, 2), 
                target_bins_masked, 
                ignore_index=-100,
                reduction='none' 
            )
            
            # 2. Continuous Sequence Separation Math
            seq_idx = torch.arange(L, device=raw_disto_loss.device)
            seq_separation = torch.abs(seq_idx.unsqueeze(0) - seq_idx.unsqueeze(1)).float() 
            seq_separation = seq_separation.unsqueeze(0) # [1, L, L]
            
            # Base sequence weight smoothly scales from 1.0 (local) to 5.0 (long-range)
            scaled_sep = torch.clamp(seq_separation / 100.0, min=0.0, max=1.0)
            seq_weight = 1.0 + (5.0 - 1.0) * scaled_sep
            
            # 3. Surgical Masking (Fixing the Background Avalanche)
            is_contact = target_pdists < 8.0  # Boolean mask of actual physical structure
            
            # Suppress the 88,000 empty background pixels so they don't dominate the loss
            weight_matrix = torch.full_like(raw_disto_loss, 1.0) 
            
            # For true contacts, use the continuous sequence weight, multiplied by a boost!
            # - A local helix contact gets ~ 1.0 * 5.0 = 5.0 weight
            # - A distant tertiary contact gets ~ 5.0 * 5.0 = 25.0 weight
            contact_weights = seq_weight * 5.0
            weight_matrix = torch.where(is_contact, contact_weights, weight_matrix)
            
            # 4. Apply weights and calculate mean
            weighted_loss = raw_disto_loss * weight_matrix
            
            valid_pixels = (target_bins_masked != -100)
            loss_disto = weighted_loss[valid_pixels].mean()
            
    # Combine all structural representations
    total_loss = (mse_trig + 
                  (lambda_dist * mse_dist) + 
                  (lambda_3d * loss_3d) + 
                  (lambda_ss * loss_ss) + 
                  (lambda_disto * loss_disto))
    
    return total_loss, mse_trig, mse_dist, loss_3d, loss_ss, loss_disto, target_pdists