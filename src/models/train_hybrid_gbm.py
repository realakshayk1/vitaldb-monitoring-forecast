from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from src.nn.dataset import VitalSeqDataset
from src.nn.model import TemporalCNN


OUT_DIR = Path("reports/hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("models/nn")


def metrics(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y, p)),
    }


def load_neural(arch: str, in_channels: int, device: str):
    ckpt = MODEL_DIR / f"{arch}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Train {arch} first.")

    if arch == "cnn":
        model = TemporalCNN(in_channels=in_channels, dropout=0.0).to(device)
    elif arch == "tcn":
        from src.nn.model_tcn import TemporalTCN
        model = TemporalTCN(in_channels=in_channels, dropout=0.0).to(device)
    else:
        raise ValueError(arch)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def embed_split(model, split: str, device: str, batch_size: int = 256):
    ds = VitalSeqDataset(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    Zs, Ys = [], []
    for X, y in loader:
        X = X.to(device)
        z = model.forward_features(X)  # (B, d)
        Zs.append(z.detach().cpu().numpy())
        Ys.append(y.detach().cpu().numpy())

    Z = np.concatenate(Zs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    return Z, Y


def main():
    arch = "cnn"        # change to "tcn" after you train it
    in_channels = 9     # must match your dataset output
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_neural(arch, in_channels, device)

    Z_train, y_train = embed_split(model, "train", device)
    Z_val, y_val = embed_split(model, "val", device)
    Z_test, y_test = embed_split(model, "test", device)

    train_ds = lgb.Dataset(Z_train, label=y_train)
    val_ds = lgb.Dataset(Z_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
    }

    gbm = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)],
    )

    val_p = gbm.predict(Z_val)
    test_p = gbm.predict(Z_test)

    val_m = metrics(y_val, val_p)
    test_m = metrics(y_test, test_p)

    payload = {
        "model": f"{arch}_embed+GBM",
        "arch": arch,
        "in_channels": in_channels,
        "val": val_m,
        "test": test_m,
        "params": params,
        "best_iteration": int(gbm.best_iteration) if gbm.best_iteration else None,
    }

    out_path = OUT_DIR / f"hybrid_{arch}_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"✅ Saved: {out_path}")
    print(payload)


if __name__ == "__main__":
    main()
