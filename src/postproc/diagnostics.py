import numpy as np


def rmsd(pred_coords, target_coords):
    pred = np.asarray(pred_coords)
    targ = np.asarray(target_coords)
    if pred.shape != targ.shape:
        raise ValueError("pred_coords and target_coords must have same shape")
    return float(np.sqrt(np.mean((pred - targ) ** 2)))


def lever_arm_ratio(local_error, global_error):
    local = float(local_error)
    global_e = float(global_error)
    if local <= 1e-12:
        return float("inf") if global_e > 0 else 1.0
    return global_e / local
