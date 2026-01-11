from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.nn.dataset import VitalSeqDataset
from src.nn.model import TemporalCNN
from src.nn.mc_dropout import mc_predict_proba

OUT = Path("reports/policy")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("models/nn/cnn_best.pt")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/nn/cnn_best.pt not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    test_ds = VitalSeqDataset("test")
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TemporalCNN(in_channels=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    means = []
    stds = []
    ys = []

    for X, y in loader:
        X = X.to(device)
        mean_p, std_p = mc_predict_proba(model, X, n_samples=20)
        means.append(mean_p)
        stds.append(std_p)
        ys.append(y.numpy())

    mean_p = np.concatenate(means)
    std_p = np.concatenate(stds)
    y = np.concatenate(ys)

    idx = test_ds.idx.copy().reset_index(drop=True)
    idx["p_mean"] = mean_p
    idx["p_std"] = std_p
    idx["y"] = y.astype(int)

    out_path = OUT / "cnn_test_uncertainty.parquet"
    idx.to_parquet(out_path, index=False)

    print("\n✅ saved:", out_path)
    print("p_mean range:", float(idx["p_mean"].min()), float(idx["p_mean"].max()))
    print("p_std  range:", float(idx["p_std"].min()), float(idx["p_std"].max()))
    print("avg p_std:", float(idx["p_std"].mean()))

if __name__ == "__main__":
    main()
