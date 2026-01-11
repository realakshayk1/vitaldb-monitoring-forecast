from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

LABELED_DIR = Path("data/processed/labeled_cases")
INP = Path("reports/policy/cnn_test_uncertainty.parquet")
OUT_DIR = Path("reports/policy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_SEC = 300
COOLDOWN_SEC = 120

def get_onsets(caseid: int) -> np.ndarray:
    df = pd.read_parquet(LABELED_DIR / f"{caseid:06d}.parquet", columns=["time_sec", "event_onset"])
    onsets = df.loc[df["event_onset"] == 1, "time_sec"].to_numpy(dtype=int)
    return np.sort(onsets)

def simulate_case(times, p_mean, p_std, onsets, tau, sigma_max):
    last_alert = -10**12
    claimed = np.zeros(len(onsets), dtype=bool)
    alerts = []

    for t, pm, ps in zip(times, p_mean, p_std):
        if pm < tau:
            continue
        if ps > sigma_max:
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
            "p_mean": float(pm),
            "p_std": float(ps),
            "matched_event": matched,
            "t_onset": onset_t,
            "lead_time_sec": lead,
        })

    return pd.DataFrame(alerts), int(claimed.sum()), int(len(onsets))

def summarize(all_alerts, caught, total_events, total_duration_sec):
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
    df = pd.read_parquet(INP).sort_values(["caseid", "t_end"])

    total_duration_sec = float(df.groupby("caseid")["t_end"].max().sum())

    # choose a few tau values around your chosen operating point
    taus = [0.45, 0.48, 0.50, 0.52]

    # choose sigma cutoffs via quantiles of p_std
    sigmas = df["p_std"].quantile([0.5, 0.7, 0.85, 0.95]).to_list()

    rows = []

    for tau in taus:
        for sigma_max in sigmas:
            alerts_rows = []
            caught_total = 0
            events_total = 0

            for caseid, g in df.groupby("caseid"):
                times = g["t_end"].to_numpy(dtype=int)
                p_mean = g["p_mean"].to_numpy(dtype=float)
                p_std = g["p_std"].to_numpy(dtype=float)

                onsets = get_onsets(int(caseid))

                alerts, caught, total_events = simulate_case(times, p_mean, p_std, onsets, tau, sigma_max)
                caught_total += caught
                events_total += total_events

                if len(alerts):
                    alerts["caseid"] = int(caseid)
                    alerts["tau"] = float(tau)
                    alerts["sigma_max"] = float(sigma_max)
                    alerts_rows.append(alerts)

            all_alerts = pd.concat(alerts_rows, ignore_index=True) if alerts_rows else pd.DataFrame(
                columns=["caseid","tau","sigma_max","t_alert","p_mean","p_std","matched_event","t_onset","lead_time_sec"]
            )

            s = summarize(all_alerts, caught_total, events_total, total_duration_sec)
            s["tau"] = tau
            s["sigma_max"] = float(sigma_max)
            rows.append(s)

    summary = pd.DataFrame(rows).sort_values(["tau","sigma_max"])
    summary.to_csv(OUT_DIR / "cnn_uncertainty_policy_summary.csv", index=False)

    print("\n✅ Uncertainty-gated policy complete")
    print(summary)

if __name__ == "__main__":
    main()
