"""Fixed-width causal GRU with separate scale and quality-proxy heads.

This module grants no action authority. Quality means predicted next observed
matched support, not a calibrated probability of wrong correspondence.
"""
import torch
from torch import nn


class ActionGRU(nn.Module):
    def __init__(self, extra_width, hidden=32):
        super().__init__()
        self.projection = nn.Linear(34 + extra_width, hidden)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.scale_head = nn.Linear(hidden, 1)
        self.quality_head = nn.Linear(hidden, 1)

    def forward(self, x, state=None):
        z, state = self.encoder(self.projection(x), state)
        return torch.cat((self.scale_head(z), self.quality_head(z)), -1), state.detach()


def predict(model, x, chunk=64):
    state = None
    values = []
    with torch.no_grad():
        for a in range(0, len(x), chunk):
            output, state = model(x[None, a:a+chunk], state)
            values.append(output[0].sigmoid())
    return torch.cat(values).cpu().numpy()
