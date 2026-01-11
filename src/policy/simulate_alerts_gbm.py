from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/windows.parquet")
LABELED_DIR = Path("data/processed/labeled_cases")
MODEL_PATH = Path("models/gbm.txt")
OUT_DIR = Path("reports/policy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_SEC = 300
COOLDOWN_SEC = 120
EXCLUDE = {"caseid", "time_sec", "y_h5m"}

def split_cases(df: pd.DataFrame):
    caseids = df["caseid"].unique()
    train_cases, temp_cases = train_test_split(caseids, train_size=0.6, random_state=42)
    val_cases, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)
    return set(train_cases), set(val_cases), set(test_cases)

def get_onsets(caseid: int) -> np.ndarray:
    df = pd.read_parquet(LABELED_DIR / f"{caseid:06d}.parquet", columns=["time_sec", "event_onset"])
    onsets = df.loc[df["event_onset"] == 1, "time_sec"].to_numpy(dtype=int)
    return np.sort(onsets)

def simulate_case(times: np.ndarray, probs: np.ndarray, onsets: np.ndarray, tau: float):
    last_alert = -10**12
    claimed = np.zeros(len(onsets), dtype=bool)
    alerts = []

    for t, p in zip(times, probs):
        if p < tau:
            continue
        if (t - last_alert) < COOLDOWN_SEC:
            continue

        last_alert = t

        j = np.searchsorted(onsets, t, side="left")
        matched = 0
        lead = None
        onset_t = None

        while j < len(onsets) and onsets[j] <= t + HORIZON_SEC:
            if not claimed[j]:
                claimed[j] = True
                matched = 1
                onset_t = int(onsets[j])
                lead = int(onsets[j] - t)
                break
            j += 1

        alerts.append({
            "t_alert": int(t),
            "p": float(p),
            "matched_event": matched,
            "t_onset": onset_t,
            "lead_time_sec": lead,
        })

    return pd.DataFrame(alerts), int(claimed.sum()), int(len(onsets))

def summarize(all_alerts: pd.DataFrame, caught: int, total_events: int, total_duration_sec: float):
    alerts = len(all_alerts)
    alerts_per_hour = alerts / (total_duration_sec / 3600.0) if total_duration_sec > 0 else np.nan
    precision = float(all_alerts["matched_event"].mean()) if alerts > 0 else np.nan
    recall = caught / total_events if total_events > 0 else np.nan

    lead = all_alerts.loc[all_alerts["matched_event"] == 1, "lead_time_sec"].dropna()
    med_lead = float(np.median(lead)) if len(lead) else np.nan
    p90_lead = float(np.percentile(lead, 90)) if len(lead) else np.nan

    return {
        "alerts": alerts,
        "alerts_per_hour": float(alerts_per_hour),
        "precision": precision,
        "event_recall": float(recall),
        "median_lead_time_sec": med_lead,
        "p90_lead_time_sec": p90_lead,
    }

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/gbm.txt not found. Re-run train_gbm with save_model().")

    df = pd.read_parquet(DATA).sort_values(["caseid", "time_sec"])
    _, _, test_cases = split_cases(df)
    test_df = df[df["caseid"].isin(test_cases)].copy()

    # total monitored duration across cases ~ max time per case
    total_duration_sec = float(test_df.groupby("caseid")["time_sec"].max().sum())

    model = lgb.Booster(model_file=str(MODEL_PATH))
    X = test_df.drop(columns=list(EXCLUDE))
    test_df["p"] = model.predict(X)

    taus = [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]
    summary_rows = []

    for tau in taus:
        alerts_rows = []
        caught_total = 0
        events_total = 0

        for caseid, g in test_df.groupby("caseid"):
            times = g["time_sec"].to_numpy(dtype=int)
            probs = g["p"].to_numpy(dtype=float)
            onsets = get_onsets(int(caseid))

            alerts, caught, total_events = simulate_case(times, probs, onsets, tau)
            caught_total += caught
            events_total += total_events

            if len(alerts):
                alerts["caseid"] = int(caseid)
                alerts["tau"] = float(tau)
                alerts_rows.append(alerts)

        all_alerts = pd.concat(alerts_rows, ignore_index=True) if alerts_rows else pd.DataFrame(
            columns=["caseid", "tau", "t_alert", "p", "matched_event", "t_onset", "lead_time_sec"]
        )

        all_alerts.to_csv(OUT_DIR / f"gbm_alerts_tau_{tau:.1f}.csv", index=False)

        s = summarize(all_alerts, caught_total, events_total, total_duration_sec)
        s["tau"] = tau
        summary_rows.append(s)

    summary = pd.DataFrame(summary_rows).sort_values("tau")
    summary.to_csv(OUT_DIR / "gbm_policy_summary.csv", index=False)

    print("\n✅ GBM policy simulation complete")
    print(summary)

if __name__ == "__main__":
    main()

