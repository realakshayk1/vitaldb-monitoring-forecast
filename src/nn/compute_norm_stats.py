from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import json

from src.nn.split import split_caseids

LABELED_DIR = Path("data/processed/labeled_cases")
OUT_DIR = Path("data/processed/seq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS = ["hr", "spo2", "map"]

def main():
    # caseids present in labeled dir
    caseids = np.array([int(p.stem) for p in LABELED_DIR.glob("*.parquet")])
    train_cases, val_cases, test_cases = split_caseids(caseids, seed=42)

    # accumulate sum/count/sumsq over observed values
    sums = {s: 0.0 for s in SIGNALS}
    sumsq = {s: 0.0 for s in SIGNALS}
    counts = {s: 0 for s in SIGNALS}

    for cid in train_cases:
        df = pd.read_parquet(LABELED_DIR / f"{cid:06d}.parquet", columns=SIGNALS)
        for s in SIGNALS:
            x = df[s].to_numpy(dtype=float)
            m = ~np.isnan(x)
            xv = x[m]
            if xv.size == 0:
                continue
            sums[s] += float(xv.sum())
            sumsq[s] += float((xv * xv).sum())
            counts[s] += int(xv.size)

    stats = {}
    for s in SIGNALS:
        mean = sums[s] / max(counts[s], 1)
        var = (sumsq[s] / max(counts[s], 1)) - mean * mean
        std = float(np.sqrt(max(var, 1e-8)))
        stats[s] = {"mean": float(mean), "std": std, "count": int(counts[s])}

    out = {
        "seed": 42,
        "train_cases": train_cases.tolist(),
        "val_cases": val_cases.tolist(),
        "test_cases": test_cases.tolist(),
        "stats": stats,
    }

    (OUT_DIR / "norm_stats.json").write_text(json.dumps(out, indent=2))
    print("\n✅ norm stats saved to data/processed/seq/norm_stats.json")
    print(stats)

if __name__ == "__main__":
    main()
