from __future__ import annotations
import numpy as np

def split_caseids(caseids: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    caseids = np.array(sorted(caseids))
    rng.shuffle(caseids)

    n = len(caseids)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train = caseids[:n_train]
    val = caseids[n_train:n_train + n_val]
    test = caseids[n_train + n_val:]
    return train, val, test
