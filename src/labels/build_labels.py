from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.labels.spec import HORIZON_SEC, MAP_THRESHOLD, MIN_LOW_DURATION_SEC

INTERIM_CASE_DIR = Path("data/interim/cases")
OUT_DIR = Path("data/processed/labeled_cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_event_onsets(map_low: np.ndarray, min_len: int) -> np.ndarray:
    """
    Given boolean array map_low[t] = True if MAP < threshold at time t,
    return boolean array onset[t] = True if a sustained low segment starts at t.

    A sustained segment is >= min_len consecutive True values.
    """
    n = len(map_low)
    onset = np.zeros(n, dtype=bool)
    if n == 0:
        return onset

    i = 0
    while i < n:
        if not map_low[i]:
            i += 1
            continue
        # start of a True run
        j = i
        while j < n and map_low[j]:
            j += 1
        run_len = j - i
        if run_len >= min_len:
            onset[i] = True
        i = j
    return onset

def build_future_event_label(onset: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    y[t] = 1 if any onset occurs in [t, t+horizon] (inclusive of t, exclusive of t+horizon+1)
    Also return t_event_onset[t] = time index of the next onset within horizon, else NaN.
    """
    n = len(onset)
    y = np.zeros(n, dtype=np.int8)
    next_onset_time = np.full(n, np.nan)

    onset_times = np.where(onset)[0]
    if onset_times.size == 0:
        return y, next_onset_time

    # For each time t, find the first onset >= t using searchsorted
    for t in range(n):
        idx = np.searchsorted(onset_times, t, side="left")
        if idx >= len(onset_times):
            continue
        t0 = onset_times[idx]
        if t0 <= t + horizon:
            y[t] = 1
            next_onset_time[t] = float(t0)

    return y, next_onset_time

def main() -> None:
    case_files = sorted(INTERIM_CASE_DIR.glob("*.parquet"))
    if not case_files:
        raise FileNotFoundError(f"No parquet files found in {INTERIM_CASE_DIR}")

    stats_rows = []

    for path in tqdm(case_files, desc="Labeling cases"):
        df = pd.read_parquet(path)

        if "map" not in df.columns:
            continue

        # Map array
        map_vals = df["map"].to_numpy(dtype=float)

        # Basic validity mask: MAP present
        map_valid = ~np.isnan(map_vals)

        # Define low MAP boolean (NaN -> False)
        map_low = (map_vals < MAP_THRESHOLD) & map_valid

        # Sustained hypotension event begins at t if low lasts >= MIN_LOW_DURATION_SEC
        onset = find_event_onsets(map_low.astype(bool), MIN_LOW_DURATION_SEC)

        # Build early-warning label
        y_h5m, t_event_onset = build_future_event_label(onset, HORIZON_SEC)

        # Attach columns
        df["map_low"] = map_low.astype(np.int8)
        df["event_onset"] = onset.astype(np.int8)
        df["y_h5m"] = y_h5m
        df["t_event_onset"] = t_event_onset

        caseid = int(path.stem)
        out_path = OUT_DIR / f"{caseid:06d}.parquet"
        df.to_parquet(out_path, index=False)

        # Stats
        n = len(df)
        n_onsets = int(onset.sum())
        pos_rate = float(y_h5m.mean()) if n > 0 else 0.0

        stats_rows.append({
            "caseid": caseid,
            "n_seconds": n,
            "n_event_onsets": n_onsets,
            "label_pos_rate": pos_rate,
            "any_event": int(n_onsets > 0),
        })

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(Path("data/processed/label_stats.csv"), index=False)

    print("\n✅ Labeling complete")
    print("Cases labeled:", len(stats))
    if len(stats) > 0:
        print("Cases with any event:", int(stats["any_event"].sum()))
        print("Median pos rate:", float(stats["label_pos_rate"].median()))
        print("Mean pos rate:", float(stats["label_pos_rate"].mean()))

if __name__ == "__main__":
    main()
