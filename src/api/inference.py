# src/api/inference.py
from __future__ import annotations

import numpy as np
import torch

from src.api.config import MC_SAMPLES


@torch.no_grad()
def mc_predict(model, x_ct: np.ndarray, device: str) -> tuple[float, float]:
    """
    x_ct: (C, T) float32
    returns (p_mean, p_std)
    """
    model.train()  # keep dropout active for MC Dropout

    X = torch.from_numpy(x_ct).unsqueeze(0).to(device)  # (1, C, T)

    ps = []
    for _ in range(MC_SAMPLES):
        logits = model(X)  # could be shape (1,) or (1,1) depending on your model
        if logits.ndim > 1:
            logits = logits.view(-1)
        p = torch.sigmoid(logits)[0].item()
        ps.append(p)

    ps = np.asarray(ps, dtype=np.float32)
    return float(ps.mean()), float(ps.std())
