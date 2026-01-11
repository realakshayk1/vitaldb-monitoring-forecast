from __future__ import annotations
import pandas as pd

P = "reports/drift/drift_report.csv"
K = 3  # require K consecutive buckets

def main():
    df = pd.read_csv(P)

    df["flag"] = df["retrain_recommended"].astype(int)
    # rolling sum of last K flags
    df["consecutive_flags"] = df["flag"].rolling(K).sum()
    df["retrain_triggered"] = df["consecutive_flags"] >= K

    out = "reports/drift/retrain_trigger.csv"
    df.to_csv(out, index=False)

    print("\n✅ retrain trigger saved:", out)
    print(df[df["retrain_triggered"]].tail(10)[
        ["mon_case_min","mon_case_max","data_drift","perf_drift","retrain_triggered"]
    ])

if __name__ == "__main__":
    main()
