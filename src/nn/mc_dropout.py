from __future__ import annotations
import numpy as np
import torch

@torch.no_grad()
def mc_predict_proba(model, X, n_samples: int = 20):
    """
    MC Dropout predictive distribution.
    Returns:
      mean_p: (B,)
      std_p: (B,)
    """
    model.train()  # keep dropout active
    ps = []
    for _ in range(n_samples):
        logits = model(X)
        p = torch.sigmoid(logits)
        ps.append(p.detach().cpu().numpy())

    ps = np.stack(ps, axis=0)  # (S, B)
    return ps.mean(axis=0), ps.std(axis=0)
