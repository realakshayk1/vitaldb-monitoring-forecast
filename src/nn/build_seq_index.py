from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.features.spec import LOOKBACK_SEC, STRIDE_SEC

LABELED_DIR = Path("data/processed/labeled_cases")
OUT_DIR = Path("data/processed/seq")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    rows = []
    files = sorted(LABELED_DIR.glob("*.parquet"))

    for p in tqdm(files, desc="Indexing cases"):
        df = pd.read_parquet(p, columns=["time_sec", "y_h5m"])
        caseid = int(p.stem)
        n = len(df)

        for t in range(LOOKBACK_SEC, n, STRIDE_SEC):
            rows.append({
                "caseid": caseid,
                "t_end": int(df.iloc[t]["time_sec"]),
                "row_t": int(t),          # row index to slice quickly
                "y": int(df.iloc[t]["y_h5m"]),
            })

    idx = pd.DataFrame(rows)
    idx.to_parquet(OUT_DIR / "seq_index.parquet", index=False)

    print("\n✅ seq index built")
    print("windows:", len(idx))
    print("pos rate:", idx["y"].mean())

if __name__ == "__main__":
    main()
