from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.nn.dataset import VitalSeqDataset
from src.nn.model import TemporalCNN
from src.nn.model_tcn import TemporalTCN  # if you have this

OUT_DIR = Path("data/processed/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_DIR = Path("data/processed/seq")
SPLIT_JSON = SEQ_DIR / "norm_stats.json"

def load_splits() -> dict:
    return json.loads(SPLIT_JSON.read_text())

def build_model(arch: str, in_channels: int, ckpt_path: Path, device: str):
    if arch == "cnn":
        model = TemporalCNN(in_channels=in_channels).to(device)
    elif arch == "tcn":
        model = TemporalTCN(in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown arch={arch}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def extract_for_split(split: str, arch: str, in_channels: int, ckpt_path: Path) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = VitalSeqDataset(split)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    model = build_model(arch, in_channels, ckpt_path, device)

    # VitalSeqDataset.idx has: caseid, t_end, row_t, y
    idx = ds.idx.copy().reset_index(drop=True)

    embs = []
    for X, _y in loader:
        X = X.to(device)
        if hasattr(model, "encode"):
            z = model.encode(X)          # CNN: (B, 64)
        else:
            # fallback for models without encode(): take pooled representation
            z = model.stem(X)
            z = model.blocks(z)
            z = model.pool(z).squeeze(-1)
        embs.append(z.detach().cpu().numpy())

    Z = np.concatenate(embs, axis=0)

    out = idx[["caseid", "t_end", "y"]].rename(columns={"t_end": "time_sec"})
    for j in range(Z.shape[1]):
        out[f"emb_{j:03d}"] = Z[:, j].astype(np.float32)

    out_path = OUT_DIR / f"{arch}_{split}_embeddings.parquet"
    out.to_parquet(out_path, index=False)
    print(f"✅ Saved embeddings: {out_path}  shape={out.shape}")
    return out_path

def main():
    # Choose which NN you want to embed from
    arch = "cnn"  # change to "tcn" if you want
    in_channels = 9
    ckpt_path = Path(f"models/nn/{arch}_best.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    for split in ["train", "val", "test"]:
        extract_for_split(split, arch, in_channels, ckpt_path)

if __name__ == "__main__":
    main()
