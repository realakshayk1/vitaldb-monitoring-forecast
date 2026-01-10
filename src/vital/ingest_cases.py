from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import vitaldb
from tqdm import tqdm

INTERIM_DIR = Path("data/interim")
CASE_DIR = INTERIM_DIR / "cases"
CASE_DIR.mkdir(parents=True, exist_ok=True)

HR = "Solar8000/HR"
SPO2 = "Solar8000/PLETH_SPO2"
MAP_ART = "Solar8000/ART_MBP"
MAP_NIBP = "Solar8000/NIBP_MBP"

FFILL_LIMIT = 10  # seconds
MIN_GOOD_SEGMENT_SECONDS = 30 * 60  # 30 minutes
MIN_NONMISSING_IN_SEGMENT = {
    "hr": 0.50,
    "spo2": 0.30,
    "map": 0.70,   # NIBP can be very sparse
}

def to_1d(x) -> np.ndarray:
    """Convert vitaldb.load_case output to 1D float array (with NaNs)."""
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"Unexpected shape: {arr.shape}")
    return arr.astype(float)

def load_track_1hz(caseid: int, track: str) -> np.ndarray | None:
    try:
        x = vitaldb.load_case(caseid, track, interval=1)
        if x is None:
            return None
        arr = to_1d(x)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None

def best_segment_nonmissing_frac(series: pd.Series, seg_len: int) -> float:
    """
    Compute the maximum non-missing fraction across any contiguous seg_len window.
    This prevents sparse-but-valid signals from being discarded.
    """
    if len(series) < seg_len:
        return float(series.notna().mean())
    # rolling mean of non-missing indicator
    nm = series.notna().astype(float)
    roll = nm.rolling(seg_len, min_periods=seg_len).mean()
    return float(roll.max()) if roll.notna().any() else 0.0

def ingest_one(caseid: int) -> tuple[pd.DataFrame | None, dict | None]:
    hr = load_track_1hz(caseid, HR)
    spo2 = load_track_1hz(caseid, SPO2)
    if hr is None or spo2 is None:
        return None, None

    # Prefer ART MAP; fallback to NIBP MAP
    map_arr = load_track_1hz(caseid, MAP_ART)
    map_source = MAP_ART
    # if map_arr is None:
    #     map_arr = load_track_1hz(caseid, MAP_NIBP)
    #     map_source = MAP_NIBP
    if map_arr is None:
        return None, None

    max_len = int(max(len(hr), len(spo2), len(map_arr)))
    df = pd.DataFrame({
        "time_sec": np.arange(max_len, dtype=int),
        "hr": pd.Series(hr).reindex(range(max_len)),
        "spo2": pd.Series(spo2).reindex(range(max_len)),
        "map": pd.Series(map_arr).reindex(range(max_len)),
    })

    # forward-fill short gaps
    df[["hr", "spo2", "map"]] = df[["hr", "spo2", "map"]].ffill(limit=FFILL_LIMIT)
    df["map_source"] = map_source

    duration_sec = int(df["time_sec"].iloc[-1]) if len(df) else 0

    # Segment-based QC (max non-missing in any 30-min segment)
    hr_best = best_segment_nonmissing_frac(df["hr"], MIN_GOOD_SEGMENT_SECONDS)
    spo2_best = best_segment_nonmissing_frac(df["spo2"], MIN_GOOD_SEGMENT_SECONDS)
    map_best = best_segment_nonmissing_frac(df["map"], MIN_GOOD_SEGMENT_SECONDS)

    if hr_best < MIN_NONMISSING_IN_SEGMENT["hr"]:
        return None, None
    if spo2_best < MIN_NONMISSING_IN_SEGMENT["spo2"]:
        return None, None
    if map_best < MIN_NONMISSING_IN_SEGMENT["map"]:
        return None, None

    meta = {
        "caseid": caseid,
        "duration_sec": duration_sec,
        "map_source": map_source,
        "hr_best_nonmissing_30m": hr_best,
        "spo2_best_nonmissing_30m": spo2_best,
        "map_best_nonmissing_30m": map_best,
        "hr_overall_nonmissing": float(df["hr"].notna().mean()),
        "spo2_overall_nonmissing": float(df["spo2"].notna().mean()),
        "map_overall_nonmissing": float(df["map"].notna().mean()),
    }
    return df, meta

def main() -> None:
    caseids = [int(x.strip()) for x in (INTERIM_DIR / "caseids_v1.txt").read_text().splitlines()]
    manifest_rows = []

    # Clear old outputs (optional safety)
    # for p in CASE_DIR.glob("*.parquet"):
    #     p.unlink()

    for caseid in tqdm(caseids, desc="Ingesting cases"):
        df, meta = ingest_one(caseid)
        if df is None:
            continue
        out_path = CASE_DIR / f"{caseid:06d}.parquet"
        df.to_parquet(out_path, index=False)
        manifest_rows.append(meta)

    manifest = pd.DataFrame(manifest_rows)

    print("\n✅ Ingestion complete")
    print("cases saved:", len(manifest))

    if len(manifest) > 0:
        manifest.to_csv(INTERIM_DIR / "case_manifest_ingested.csv", index=False)
        print("map source counts:")
        print(manifest["map_source"].value_counts())
        print("\nBest 30-min non-missing summary:")
        print(manifest[["hr_best_nonmissing_30m","spo2_best_nonmissing_30m","map_best_nonmissing_30m"]].describe())
    else:
        print("⚠️ No cases passed QC. Next move: drop segment thresholds further or restrict to ART_MBP-only.")

if __name__ == "__main__":
    main()
