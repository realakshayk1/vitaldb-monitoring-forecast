from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.nn.dataset import VitalSeqDataset
from src.nn.model import TemporalCNN
from src.nn.model_tcn import TemporalTCN

LABELED_DIR = Path("data/processed/labeled_cases")
SEQ_DIR = Path("data/processed/seq")
OUT_DIR = Path("reports/policy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCH = "cnn"  # change to "tcn" when needed
MODEL_PATH = Path(f"models/nn/{ARCH}_best.pt")
CAL_PATH   = Path(f"models/nn/{ARCH}_calibration.json")

def load_temperature() -> float:
    if not CAL_PATH.exists():
        return 1.0
    obj = json.loads(CAL_PATH.read_text())
    return float(obj.get("temperature", 1.0))

HORIZON_SEC = 300
COOLDOWN_SEC = 120

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
        lead = np.nan
        onset_t = np.nan


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
        raise FileNotFoundError(f"models/nn/{ARCH}_best.pt not found. Finish Step 8A training first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    T = load_temperature()
    print(f"Using calibration temperature T={T:.3f}")


    # Load test dataset + its index (so we have caseid, t_end aligned to predictions)
    test_ds = VitalSeqDataset("test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    if ARCH == "cnn":
        model = TemporalCNN(in_channels=9).to(device)
    elif ARCH == "tcn":
        model = TemporalTCN(in_channels=9).to(device)
    else:
        raise ValueError(f"Unknown ARCH={ARCH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for X, _y in test_loader:
            X = X.to(device)
            logits = model(X)
            p = torch.sigmoid(logits / T).cpu().numpy()
            probs.append(p)

    probs = np.concatenate(probs)
    idx = test_ds.idx.copy().reset_index(drop=True)

    # Align predictions with index rows
    idx["p"] = probs

    # approximate total monitored duration across test cases
    total_duration_sec = float(idx.groupby("caseid")["t_end"].max().sum())

    taus = [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]
    summary_rows = []

    for tau in taus:
        alerts_rows = []
        caught_total = 0
        events_total = 0

        for caseid, g in idx.groupby("caseid"):
            times = g["t_end"].to_numpy(dtype=int)
            p_case = g["p"].to_numpy(dtype=float)
            onsets = get_onsets(int(caseid))

            alerts, caught, total_events = simulate_case(times, p_case, onsets, tau)
            caught_total += caught
            events_total += total_events

            if len(alerts):
                alerts["caseid"] = int(caseid)
                alerts["tau"] = float(tau)
                alerts_rows.append(alerts)

        all_alerts = pd.concat(alerts_rows, ignore_index=True) if alerts_rows else pd.DataFrame(
            columns=["caseid", "tau", "t_alert", "p", "matched_event", "t_onset", "lead_time_sec"]
        )

        all_alerts.to_csv(OUT_DIR / f"cnn_alerts_tau_{tau:.2f}.csv", index=False)

        s = summarize(all_alerts, caught_total, events_total, total_duration_sec)
        s["tau"] = tau
        summary_rows.append(s)

    summary = pd.DataFrame(summary_rows).sort_values("tau")
    summary.to_csv(OUT_DIR / "cnn_policy_summary.csv", index=False)

    print("\n✅ CNN policy simulation complete")
    print(summary)

if __name__ == "__main__":
    main()
