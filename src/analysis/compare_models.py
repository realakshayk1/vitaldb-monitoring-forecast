from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POLICY_DIR = Path("reports/policy")
OUT_DIR = Path("reports/compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column that exists among candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@dataclass
class PolicyCols:
    tau: str
    precision: Optional[str]
    recall: Optional[str]
    f1: Optional[str]
    alerts_per_hour: Optional[str]
    alert_rate: Optional[str]
    lead_time: Optional[str]


def infer_policy_cols(df: pd.DataFrame) -> PolicyCols:
    tau = _find_col(df, ["tau", "threshold", "prob_threshold"])
    if tau is None:
        raise ValueError("Could not find a threshold column (expected tau/threshold/prob_threshold).")

    precision = _find_col(df, ["precision", "prec", "ppv"])
 
    recall = _find_col(df, ["event_recall", "recall", "tpr", "sensitivity"])

    f1 = _find_col(df, ["f1", "f1_score"])

    alerts_per_hour = _find_col(df, ["alerts_per_hour", "alerts_hr", "alerts_per_hr"])
    alert_rate = _find_col(df, ["alert_rate", "alerts_rate", "alarm_rate", "fraction_alerts", "alert_fraction"])

    lead_time = _find_col(df, [
        "median_lead_time_sec", "median_lead_time", "lead_time_median", "median_lead_time_s",
        "p50_lead_time_sec", "lead_time_sec", "lead_time", "leadtime"
    ])

    return PolicyCols(
        tau=tau,
        precision=precision,
        recall=recall,
        f1=f1,
        alerts_per_hour=alerts_per_hour,
        alert_rate=alert_rate,
        lead_time=lead_time
    )


def compute_f1(df: pd.DataFrame, cols: PolicyCols) -> pd.DataFrame:
    if cols.f1 is None and cols.precision and cols.recall:
        df["f1"] = 2 * df[cols.precision] * df[cols.recall] / (df[cols.precision] + df[cols.recall] + 1e-12)
        cols.f1 = "f1"
    return df


def pick_operating_point(
    df: pd.DataFrame,
    cols: PolicyCols,
    *,
    mode: str = "max_f1",
    target_recall: float = 0.90,
    max_alerts_per_hour: Optional[float] = None
) -> pd.Series:
    """
    Choose a row representing an operating point.
    - mode=max_f1: pick highest F1
    - mode=recall_then_min_alerts: pick recall>=target then minimize alerts_per_hour (or alert_rate)
    """
    d = df.copy()
    d = d.sort_values(cols.tau)

    if mode == "max_f1":
        if cols.f1 is None:
            raise ValueError("No F1 available and cannot compute it (need precision+recall).")
        idx = d[cols.f1].idxmax()
        return d.loc[idx]

    if mode == "recall_then_min_alerts":
        if cols.recall is None:
            raise ValueError("Need recall column for recall_then_min_alerts mode.")
        d = d[d[cols.recall] >= target_recall].copy()
        if d.empty:
            # fall back to max recall
            idx = df[cols.recall].idxmax() if cols.recall else df.index[0]
            return df.loc[idx]

        if max_alerts_per_hour is not None and cols.alerts_per_hour:
            d = d[d[cols.alerts_per_hour] <= max_alerts_per_hour].copy()
            if d.empty:
                # if budget impossible, just pick min alerts among those >= target recall
                d = df[df[cols.recall] >= target_recall].copy()

        if cols.alerts_per_hour:
            idx = d[cols.alerts_per_hour].idxmin()
            return d.loc[idx]
        if cols.alert_rate:
            idx = d[cols.alert_rate].idxmin()
            return d.loc[idx]
        # otherwise just pick smallest tau
        return d.iloc[0]

    raise ValueError(f"Unknown mode: {mode}")


def summarize_model(name: str, summary_path: Path) -> Tuple[str, pd.DataFrame, PolicyCols]:
    df = _read_csv(summary_path)
    cols = infer_policy_cols(df)

    # coerce numerics on the columns we care about
    numeric_cols = [cols.tau]
    for c in [cols.precision, cols.recall, cols.f1, cols.alerts_per_hour, cols.alert_rate, cols.lead_time]:
        if c:
            numeric_cols.append(c)
    df = _coerce_numeric(df, numeric_cols)

    df = compute_f1(df, cols)
    return name, df, cols


def plot_tradeoffs(models: List[Tuple[str, pd.DataFrame, PolicyCols]]) -> None:
    # 1) Precision/Recall vs tau
    plt.figure()
    for name, df, cols in models:
        if cols.recall:
            plt.plot(df[cols.tau], df[cols.recall], label=f"{name} recall")
        if cols.precision:
            plt.plot(df[cols.tau], df[cols.precision], label=f"{name} precision")
    plt.xlabel("tau")
    plt.ylabel("metric")
    plt.title("Precision/Recall vs threshold (tau)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "precision_recall_vs_tau.png", dpi=150)
    plt.close()

    # 2) Alerts vs recall (operational curve)
    plt.figure()
    for name, df, cols in models:
        x = None
        if cols.alerts_per_hour:
            x = df[cols.alerts_per_hour]
            xlabel = "alerts/hour"
        elif cols.alert_rate:
            x = df[cols.alert_rate]
            xlabel = "alert rate"
        else:
            continue

        if cols.recall:
            plt.plot(x, df[cols.recall], label=name)
    plt.xlabel(xlabel)
    plt.ylabel("recall")
    plt.title("Operational tradeoff: recall vs alert workload")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "recall_vs_alerts.png", dpi=150)
    plt.close()

    # 3) Lead time vs alert workload (if available)
    any_lead = any(cols.lead_time is not None for _, _, cols in models)
    if any_lead:
        plt.figure()
        for name, df, cols in models:
            if cols.lead_time is None:
                continue
            if cols.alerts_per_hour:
                x = df[cols.alerts_per_hour]
                xlabel = "alerts/hour"
            elif cols.alert_rate:
                x = df[cols.alert_rate]
                xlabel = "alert rate"
            else:
                continue
            plt.plot(x, df[cols.lead_time], label=name)
        plt.xlabel(xlabel)
        plt.ylabel("lead time")
        plt.title("Early detection: lead time vs alert workload")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "lead_time_vs_alerts.png", dpi=150)
        plt.close()


def main():
    # You provided these exist
    paths = [
        ("GBM", POLICY_DIR / "gbm_policy_summary.csv"),
        ("CNN", POLICY_DIR / "cnn_policy_summary.csv"),
        ("CNN+Uncertainty", POLICY_DIR / "cnn_uncertainty_policy_summary.csv"),
    ]

    models: List[Tuple[str, pd.DataFrame, PolicyCols]] = []
    for name, p in paths:
        if p.exists():
            models.append(summarize_model(name, p))
        else:
            print(f"⚠️ Missing: {p}")

    if not models:
        raise RuntimeError("No policy summary CSVs found to compare.")

    # Print “best operating points” under two common decision rules
    rows = []
    for name, df, cols in models:
        # max F1
        best_f1 = None
        try:
            best_f1 = pick_operating_point(df, cols, mode="max_f1")
        except Exception:
            best_f1 = None

        # fixed recall then min alerts
        best_budget = None
        try:
            best_budget = pick_operating_point(df, cols, mode="recall_then_min_alerts", target_recall=0.90)
        except Exception:
            best_budget = None

        def pack(tag: str, s: Optional[pd.Series]) -> Optional[Dict]:
            if s is None:
                return None
            out = {"model": name, "selection": tag, "tau": float(s[cols.tau])}
            for k, col in [
                ("precision", cols.precision),
                ("recall", cols.recall),
                ("f1", cols.f1),
                ("alerts_per_hour", cols.alerts_per_hour),
                ("alert_rate", cols.alert_rate),
                ("lead_time", cols.lead_time),
            ]:
                if col and col in s.index:
                    v = s[col]
                    out[k] = None if pd.isna(v) else float(v)
            return out

        if best_f1 is not None:
            rows.append(pack("max_f1", best_f1))
        if best_budget is not None:
            rows.append(pack("recall>=0.90_min_alerts", best_budget))

    out_df = pd.DataFrame([r for r in rows if r is not None])
    out_path = OUT_DIR / "model_policy_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print("\n=== Operating point comparison (saved to reports/compare/model_policy_comparison.csv) ===")
    with pd.option_context("display.max_columns", 50, "display.width", 140):
        print(out_df)

    # Plot tradeoffs
    plot_tradeoffs(models)
    print(f"\n📈 Saved plots to: {OUT_DIR}")
    print(f"  - precision_recall_vs_tau.png")
    print(f"  - recall_vs_alerts.png")
    print(f"  - lead_time_vs_alerts.png (if lead_time column exists)")


if __name__ == "__main__":
    main()
