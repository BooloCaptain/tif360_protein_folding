import numpy as np


def angles_to_sincos(angles):
    """Convert angles (radians) to stacked sin/cos pairs.

    angles: array-like, shape (...)
    returns: array shape (..., 2)
    """
    a = np.array(angles)
    return np.stack([np.sin(a), np.cos(a)], axis=-1)


def trig_mse(pred_sincos, target_angles):
    """Compute MSE between predicted sin/cos pairs and target angles.

    pred_sincos: array-like shape (..., 2) or (..., 4) if containing theta and tau pairs concatenated
    target_angles: array-like shape (...) or (..., 2) matching pred structure
    Returns scalar MSE.
    """
    pred = np.array(pred_sincos)
    targ_angles = np.array(target_angles)

    # If pred has lastdim 4, assume two angle pairs concatenated
    if pred.shape[-1] == 4:
        # split into two pairs
        pred = pred.reshape(*pred.shape[:-1], 2, 2)
        # targ_angles expected shape (...,2)
        targ = np.stack([np.stack([np.sin(targ_angles[..., i]), np.cos(targ_angles[..., i])], axis=-1) for i in range(targ_angles.shape[-1])], axis=-2)
        # targ shape (...,2,2)
    elif pred.shape[-1] == 2:
        targ = np.stack([np.sin(targ_angles), np.cos(targ_angles)], axis=-1)
    else:
        raise ValueError("pred_sincos must have last dim 2 or 4")

    mse = np.mean((pred - targ) ** 2)
    return mse
