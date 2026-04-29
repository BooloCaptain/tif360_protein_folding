import numpy as np
from src.losses.trig_loss import angles_to_sincos, trig_mse


def test_angles_to_sincos_and_trig_mse():
    angles = np.array([0.0, np.pi/2])
    sc = angles_to_sincos(angles)
    assert sc.shape == (2,2)
    # test mse with perfect prediction
    mse = trig_mse(sc, angles)
    assert mse == 0.0

    # test with slight error
    pred = sc.copy()
    pred[0,0] = 0.1
    mse2 = trig_mse(pred, angles)
    assert mse2 > 0.0
