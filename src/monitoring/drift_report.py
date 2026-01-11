from __future__ import annotations
from typing import List

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

INP = Path("reports/policy/cnn_test_uncertainty.parquet")
OUT = Path("reports/drift/drift_report.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Your selected operating policy
TAU = 0.48
SIGMA_MAX = 0.03229

# Drift thresholds (reasonable defaults)
KS_P_THRESHOLD = 0.01   # small p-value = drift
AUPRC_DROP_FRAC = 0.20  # 20% drop from reference triggers

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)

def main():
    df = pd.read_parquet(INP).sort_values(["caseid", "t_end"]).reset_index(drop=True)

    # ---- 9B+ Merge raw telemetry features from windows.parquet ----
    WIN_PATH = Path("data/processed/windows.parquet")

    RAW_FEATURES = [
        "hr_mean", "hr_std", "hr_slope",
        "map_mean", "map_std", "map_slope",
        "spo2_mean",
    ]

    windows = pd.read_parquet(WIN_PATH)

    # windows uses time_sec; uncertainty uses t_end
    # keep only what we need to reduce memory
    keep_cols = ["caseid", "time_sec"] + [c for c in RAW_FEATURES if c in windows.columns]
    windows = windows[keep_cols].sort_values(["caseid", "time_sec"]).reset_index(drop=True)

    # auto-drop any missing requested features (e.g., spo2_mean)
    actual_raw_feats = [c for c in RAW_FEATURES if c in windows.columns]
    print("raw telemetry features used:", actual_raw_feats)

    # Use merge_asof to match nearest time within each caseid
    # tolerance should be <= your stride; if stride is 10s, use 10; if 30s, use 30.
    TOL_SEC = 10

    df = df.sort_values(["caseid", "t_end"]).reset_index(drop=True)
    windows = windows.sort_values(["caseid", "time_sec"]).reset_index(drop=True)

    # --- merge_asof (bulletproof): global sort by on-key, then restore order ---

    # ensure numeric
    df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
    windows["time_sec"] = pd.to_numeric(windows["time_sec"], errors="coerce")

    df = df.dropna(subset=["caseid", "t_end"]).copy()
    windows = windows.dropna(subset=["caseid", "time_sec"]).copy()

    # GLOBAL sort by merge key (required by merge_asof)
    df_global = df.sort_values("t_end", kind="mergesort").reset_index(drop=True)
    win_global = windows.sort_values("time_sec", kind="mergesort").reset_index(drop=True)

    df_merged = pd.merge_asof(
        df_global,
        win_global,
        left_on="t_end",
        right_on="time_sec",
        by="caseid",
        direction="nearest",
        tolerance=TOL_SEC,
    )

    # restore expected ordering for downstream rolling windows
    df = df_merged.sort_values(["caseid", "t_end"], kind="mergesort").reset_index(drop=True)

    missing = df["time_sec"].isna().mean()
    print(f"merge_asof missing fraction (tolerance={TOL_SEC}s): {missing:.4f}")

    # --- end merge_asof ---


    # Optional: if some rows didn't find a match (NaN time_sec), loosen tolerance once
    missing = df["time_sec"].isna().mean()
    print(f"merge_asof missing fraction (tolerance={TOL_SEC}s): {missing:.4f}")

    if missing > 0.05:
        TOL_SEC2 = 30
        df2 = pd.merge_asof(
            df.drop(columns=["time_sec"], errors="ignore"),
            windows,
            left_on="t_end",
            right_on="time_sec",
            by="caseid",
            direction="nearest",
            tolerance=TOL_SEC2,
        )
        missing2 = df2["time_sec"].isna().mean()
        print(f"merge_asof missing fraction (tolerance={TOL_SEC2}s): {missing2:.4f}")
        df = df2
    # ---- end merge ----

    # Define alert decisions for monitoring
    df["is_alert"] = (df["p_mean"] >= TAU) & (df["p_std"] <= SIGMA_MAX)

    # Choose a compact set of “monitoring features”
    # (these are derived from model outputs; later you can add real signal features too)
    features = ["p_mean", "p_std"] + actual_raw_feats

    caseids = np.array(sorted(df["caseid"].unique()))

    print("unique test cases:", len(caseids))

    # Adaptive window sizes based on available test cases
    MIN_WINDOW = 8  # keep at least 8 cases per window for stability

    n_cases = len(caseids)
    REF_CASES = max(MIN_WINDOW, n_cases // 4)
    MONITOR_CASES = max(MIN_WINDOW, n_cases // 4)
    STEP_CASES = max(3, min(10, n_cases // 10))

    # If still too small, fall back to a single split
    if REF_CASES + MONITOR_CASES > n_cases:
        REF_CASES = max(MIN_WINDOW, n_cases // 3)
        MONITOR_CASES = max(MIN_WINDOW, n_cases - REF_CASES)
        STEP_CASES = max(1, REF_CASES // 2)
    print("REF_CASES:", REF_CASES, "MONITOR_CASES:", MONITOR_CASES, "STEP_CASES:", STEP_CASES)

    rows = []
    for start in range(0, len(caseids) - (REF_CASES + MONITOR_CASES) + 1, STEP_CASES):
        ref_cases = caseids[start : start + REF_CASES]
        mon_cases = caseids[start + REF_CASES : start + REF_CASES + MONITOR_CASES]

        ref = df[df["caseid"].isin(ref_cases)]
        mon = df[df["caseid"].isin(mon_cases)]

        # Performance drift
        ref_auprc = safe_auprc(ref["y"].values, ref["p_mean"].values)
        mon_auprc = safe_auprc(mon["y"].values, mon["p_mean"].values)

        ref_auc = safe_auc(ref["y"].values, ref["p_mean"].values)
        mon_auc = safe_auc(mon["y"].values, mon["p_mean"].values)

        # Data drift (KS test per feature)
        ks_stats = {}
        ks_ps = {}
        drift_flags = []

        for f in features:
            r = ref[f].values
            m = mon[f].values
            stat, p = ks_2samp(r, m)
            ks_stats[f] = float(stat)
            ks_ps[f] = float(p)
            drift_flags.append(p < KS_P_THRESHOLD)

        any_data_drift = bool(any(drift_flags))

        # Performance drift flag (relative drop)
        perf_drift = False
        if np.isfinite(ref_auprc) and np.isfinite(mon_auprc) and ref_auprc > 0:
            perf_drift = (mon_auprc < (1.0 - AUPRC_DROP_FRAC) * ref_auprc)

        # Alert burden drift (optional but useful)
        ref_alert_rate = float(ref["is_alert"].mean())
        mon_alert_rate = float(mon["is_alert"].mean())

        row = {
        "ref_case_min": int(ref_cases.min()),
        "ref_case_max": int(ref_cases.max()),
        "mon_case_min": int(mon_cases.min()),
        "mon_case_max": int(mon_cases.max()),
        "ref_auprc": float(ref_auprc) if np.isfinite(ref_auprc) else np.nan,
        "mon_auprc": float(mon_auprc) if np.isfinite(mon_auprc) else np.nan,
        "ref_auc": float(ref_auc) if np.isfinite(ref_auc) else np.nan,
        "mon_auc": float(mon_auc) if np.isfinite(mon_auc) else np.nan,
        "ref_alert_rate": float(ref_alert_rate),
        "mon_alert_rate": float(mon_alert_rate),
        "data_drift": bool(any_data_drift),
        "perf_drift": bool(perf_drift),
        "retrain_recommended": bool(any_data_drift or perf_drift),
    }

    # ✅ add KS metrics for every drift feature (model + raw telemetry)
    for f in features:
        row[f"ks_stat_{f}"] = float(ks_stats[f])
        row[f"ks_p_{f}"] = float(ks_ps[f])

    rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(OUT, index=False)
    print("\n✅ Drift report saved:", OUT)
    print(report.tail(10))

if __name__ == "__main__":
    main()
