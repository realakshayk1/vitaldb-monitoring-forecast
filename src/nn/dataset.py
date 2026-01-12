from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.spec import LOOKBACK_SEC

LABELED_DIR = Path("data/processed/labeled_cases")
SEQ_DIR = Path("data/processed/seq")

SIGNALS = ["hr", "spo2", "map"]

class VitalSeqDataset(Dataset):
    """
    Returns:
      X: float32 tensor shaped (C, T) where C=9 (3 values + 3 masks + 3 time-since-observed)
      y: float32 scalar (0/1)
    """
    def __init__(self, split: str):
        idx = pd.read_parquet(SEQ_DIR / "seq_index.parquet")

        # load splits + normalization
        raw = (SEQ_DIR / "norm_stats.json").read_text()
        splits = json.loads(raw)

        self.stats = splits["stats"]
        split_cases = set(splits[f"{split}_cases"])

        self.idx = idx[idx["caseid"].isin(split_cases)].reset_index(drop=True)

        # simple per-process case cache (big speedup)
        self._cache_caseid = None
        self._cache_df = None

    def __len__(self):
        return len(self.idx)

    def _load_case(self, caseid: int) -> pd.DataFrame:
        if self._cache_caseid == caseid and self._cache_df is not None:
            return self._cache_df
        df = pd.read_parquet(LABELED_DIR / f"{caseid:06d}.parquet", columns=SIGNALS)
        self._cache_caseid = caseid
        self._cache_df = df
        return df

    def __getitem__(self, i: int):
        row = self.idx.iloc[i]
        caseid = int(row["caseid"])
        t = int(row["row_t"])

        df = self._load_case(caseid)

        win = df.iloc[t - LOOKBACK_SEC : t]  # length LOOKBACK_SEC
        X_list = []

        for s in SIGNALS:
            x = win[s].to_numpy(dtype=float)  # (T,)
            mask = (~np.isnan(x)).astype(np.float32)

            # time since last observed (normalized to [0,1] by window length)
            ts = np.zeros_like(mask, dtype=np.float32)
            last = -1
            T = len(mask)
            for k in range(T):
                if mask[k] == 1:
                    last = k
                    ts[k] = 0.0
                else:
                    ts[k] = (k - last) if last >= 0 else float(T)
            ts = ts / float(T)

            # normalize observed values using train stats; fill missing with 0 after normalization
            mean = float(self.stats[s]["mean"])
            std = float(self.stats[s]["std"])
            x_norm = (x - mean) / std
            x_norm = np.where(np.isnan(x_norm), 0.0, x_norm).astype(np.float32)

            X_list.append(x_norm)
            X_list.append(mask)
            X_list.append(ts)

        X = np.stack(X_list, axis=0)  # (9, T)
        y = np.float32(row["y"])

        return torch.from_numpy(X), torch.tensor(y)
