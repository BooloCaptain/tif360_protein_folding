import torch
import torch.nn.functional as F


def angles_to_sincos_torch(angles):
    # angles: (..., 2) or (...) for single angle
    a = angles
    if a.dim() == 0:
        a = a.unsqueeze(0)
    if a.dim() == 1:
        a = a.unsqueeze(-1)
    # expecting (..., n_angles)
    sin = torch.sin(a)
    cos = torch.cos(a)
    # returns (..., n_angles, 2)
    return torch.stack([sin, cos], dim=-1)


def trig_distance_loss(pred, target_angles, target_distances, lambda_dist=1.0, mask=None):
    """Compute trig MSE + lambda * distance MSE.

    pred: (B,L,5) -> [x_theta,y_theta,x_tau,y_tau,d]
    target_angles: (B,L,2) -> [theta, tau] in radians
    target_distances: (B,L)
    mask: (B,L) binary mask (1=valid)
    """
    pred_theta = pred[..., 0:2]  # (B,L,2)
    pred_tau = pred[..., 2:4]
    pred_d = pred[..., 4].squeeze(-1)

    # compute target sincos
    target_sincos = angles_to_sincos_torch(target_angles)  # (B,L,2,2)
    # target_sincos[..., i, 0] = sin, [...,i,1]=cos

    # pred vectors already normalized by head; treat as [x,y] ~ [sin,cos]
    pred_theta_sincos = pred_theta.unsqueeze(-2)  # (B,L,1,2)
    pred_tau_sincos = pred_tau.unsqueeze(-2)

    # compute MSE
    mse_theta = F.mse_loss(pred_theta_sincos, target_sincos[..., 0:1, :], reduction='none')  # (B,L,1,2)
    mse_tau = F.mse_loss(pred_tau_sincos, target_sincos[..., 1:2, :], reduction='none')

    mse_theta = mse_theta.mean(dim=(-1, -2))  # (B,L)
    mse_tau = mse_tau.mean(dim=(-1, -2))
    mse_trig = (mse_theta + mse_tau) * 0.5

    mse_dist = F.mse_loss(pred_d, target_distances, reduction='none')

    if mask is not None:
        mask = mask.float()
        mse_trig = (mse_trig * mask).sum() / (mask.sum() + 1e-8)
        mse_dist = (mse_dist * mask).sum() / (mask.sum() + 1e-8)
    else:
        mse_trig = mse_trig.mean()
        mse_dist = mse_dist.mean()

    total = mse_trig + lambda_dist * mse_dist
    return total, mse_trig, mse_dist
