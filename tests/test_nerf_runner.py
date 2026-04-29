import numpy as np
from src.postproc.nerf_runner import reconstruct_chain


def test_reconstruct_chain_small():
    # simple chain of length 5 with constant internals
    L = 5
    d = 3.8
    theta = np.pi/2
    tau = 0.0
    internals = np.tile(np.array([d, theta, tau]), (L,1))
    coords = reconstruct_chain(internals)
    assert coords.shape == (L,3)


if __name__ == '__main__':
    test_reconstruct_chain_small()
    print('ok')
