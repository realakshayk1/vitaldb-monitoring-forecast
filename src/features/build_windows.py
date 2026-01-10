from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.spec import LOOKBACK_SEC, STRIDE_SEC, SIGNALS

LABELED_DIR = Path("data/processed/labeled_cases")
OUT_PATH = Path("data/processed/windows.parquet")

def longest_nan_run(x: np.ndarray) -> int:
    max_run = run = 0
    for v in x:
        if np.isnan(v):
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def compute_features(window: pd.DataFrame) -> dict:
    feats = {}
    t = window["time_sec"].to_numpy()

    for sig in SIGNALS:
        x = window[sig].to_numpy(dtype=float)

        valid = ~np.isnan(x)
        if valid.sum() == 0:
            # all missing → mark clearly
            feats[f"{sig}_last"] = np.nan
            feats[f"{sig}_mean"] = np.nan
            feats[f"{sig}_std"] = np.nan
            feats[f"{sig}_min"] = np.nan
            feats[f"{sig}_max"] = np.nan
            feats[f"{sig}_delta"] = np.nan
            feats[f"{sig}_slope"] = np.nan
        else:
            xv = x[valid]
            tv = t[valid]

            feats[f"{sig}_last"] = xv[-1]
            feats[f"{sig}_mean"] = xv.mean()
            feats[f"{sig}_std"] = xv.std()
            feats[f"{sig}_min"] = xv.min()
            feats[f"{sig}_max"] = xv.max()
            feats[f"{sig}_delta"] = xv[-1] - xv[0]

            # slope via least squares
            if len(xv) > 1:
                slope = np.polyfit(tv - tv[0], xv, 1)[0]
            else:
                slope = 0.0
            feats[f"{sig}_slope"] = slope

        feats[f"{sig}_missing_frac"] = float(np.isnan(x).mean())
        feats[f"{sig}_max_missing_run"] = longest_nan_run(x)

    return feats

def main() -> None:
    rows = []

    case_files = sorted(LABELED_DIR.glob("*.parquet"))
    for path in tqdm(case_files, desc="Windowing cases"):
        df = pd.read_parquet(path)
        caseid = int(path.stem)

        n = len(df)
        for t in range(LOOKBACK_SEC, n, STRIDE_SEC):
            window = df.iloc[t - LOOKBACK_SEC : t]

            # label at decision time t
            y = int(df.iloc[t]["y_h5m"])

            feats = compute_features(window)
            feats["caseid"] = caseid
            feats["time_sec"] = int(df.iloc[t]["time_sec"])
            feats["y_h5m"] = y

            rows.append(feats)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_PATH, index=False)

    print("\n✅ Windowing complete")
    print("Total windows:", len(out_df))
    print("Positive rate:", out_df["y_h5m"].mean())

if __name__ == "__main__":
    main()
