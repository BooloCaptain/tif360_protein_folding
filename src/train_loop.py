import torch
from src.models.transformer import TransformerBackbone
from src.models.heads import TrigDistanceHead
from src.losses.torch_trig_loss import trig_distance_loss


def smoke_run(vocab_size=21, seq_len=10, batch=2, device='cpu'):
    device = torch.device(device)
    model = TransformerBackbone(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
    head = TrigDistanceHead(d_model=64).to(device)

    # dummy data
    tokens = torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.long, device=device)
    mask = torch.ones((batch, seq_len), dtype=torch.float32, device=device)
    # target angles theta,tau
    theta = torch.zeros((batch, seq_len), device=device)
    tau = torch.zeros((batch, seq_len), device=device)
    angles = torch.stack([theta, tau], dim=-1)
    distances = torch.ones((batch, seq_len), device=device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    model.train(); head.train()
    out = model(tokens)
    pred = head(out)
    total, mt, md = trig_distance_loss(pred, angles, distances, lambda_dist=1.0, mask=mask)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return total.item(), mt.item(), md.item()


if __name__ == '__main__':
    print(smoke_run())
