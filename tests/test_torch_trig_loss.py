import sys, os
sys.path.append(os.path.abspath('.'))
import torch
from src.losses.torch_trig_loss import trig_distance_loss


def test_trig_distance_loss_basic():
    B, L = 1, 2
    # create pred where theta=(0,pi/2), tau=(0,0), d=(1,1)
    theta = torch.tensor([[0.0, 3.1415926/2]])
    tau = torch.tensor([[0.0, 0.0]])
    angles = torch.stack([theta, tau], dim=-1)  # (B,L,2)
    # build pred matching sincos
    sincos_theta = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
    sincos_tau = torch.stack([torch.sin(tau), torch.cos(tau)], dim=-1)
    d = torch.ones((B,L,1))
    pred = torch.cat([sincos_theta, sincos_tau, d], dim=-1)
    total, mt, md = trig_distance_loss(pred, angles, d.squeeze(-1), lambda_dist=1.0)
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)


if __name__ == '__main__':
    test_trig_distance_loss_basic()
    print('ok')
