from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

P = Path("reports/policy/gbm_policy_summary.csv")

def main():
    s = pd.read_csv(P).sort_values("tau")

    plt.figure()
    plt.plot(s["alerts_per_hour"], s["event_recall"], marker="o")
    plt.xlabel("Alerts per hour (test)")
    plt.ylabel("Event recall")
    plt.title("GBM: Alert burden vs event recall")
    plt.show()

    plt.figure()
    plt.plot(s["tau"], s["precision"], marker="o", label="precision")
    plt.plot(s["tau"], s["event_recall"], marker="o", label="event_recall")
    plt.xlabel("Threshold tau")
    plt.ylabel("Metric")
    plt.title("GBM: Precision/recall vs threshold")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(s["tau"], s["median_lead_time_sec"], marker="o", label="median lead time")
    plt.plot(s["tau"], s["p90_lead_time_sec"], marker="o", label="p90 lead time")
    plt.xlabel("Threshold tau")
    plt.ylabel("Lead time (sec)")
    plt.title("GBM: Lead time vs threshold")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
