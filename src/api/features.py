# src/api/features.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.api.config import SEQ_LEN, STRIDE_SEC, CHANNEL_COLS

WIN_PATH = Path("data/processed/windows.parquet")

class WindowStore:
    """
    Simple parquet-backed feature store for demo.
    Loads windows.parquet once, then slices per request.
    """
    def __init__(self):
        if not WIN_PATH.exists():
            raise FileNotFoundError(f"{WIN_PATH} not found")

        # Load minimal columns for speed/memory
        cols = ["caseid", "time_sec"] + CHANNEL_COLS
        self.df = pd.read_parquet(WIN_PATH, columns=cols).sort_values(["caseid", "time_sec"]).reset_index(drop=True)

        # quick index for case filtering
        self.case_groups = {cid: g for cid, g in self.df.groupby("caseid", sort=False)}

    def get_sequence(self, caseid: int, t_end: int) -> np.ndarray:
        """
        Return (C, T) float32 tensor-ready array for CNN.
        Sequence ends at t_end and steps backwards by STRIDE_SEC.
        """
        if caseid not in self.case_groups:
            raise KeyError(f"caseid {caseid} not found in windows.parquet")

        g = self.case_groups[caseid]

        times = np.arange(t_end - (SEQ_LEN - 1) * STRIDE_SEC, t_end + 1, STRIDE_SEC, dtype=int)
        # exact match lookup via merge
        sub = pd.DataFrame({"time_sec": times})
        merged = sub.merge(g, on="time_sec", how="left")

        X = merged[CHANNEL_COLS].to_numpy(dtype=np.float32)  # (T, C)

        # If there are missing values (rare), simple forward/back fill then zeros
        if np.isnan(X).any():
            m = pd.DataFrame(X).ffill().bfill().to_numpy()
            X = np.nan_to_num(m, nan=0.0).astype(np.float32)

        # CNN expects (C, T)
        return X.T
