from __future__ import annotations

from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt

LABELED_DIR = Path("data/processed/labeled_cases")

def main() -> None:
    files = sorted(LABELED_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No labeled cases found.")

    sample_files = random.sample(files, k=min(3, len(files)))

    for path in sample_files:
        df = pd.read_parquet(path)
        caseid = path.stem

        t = df["time_sec"]
        m = df["map"]
        y = df["y_h5m"]
        onset = df["event_onset"]

        plt.figure()
        plt.plot(t, m, label="MAP")
        # show onset markers
        onset_t = df.loc[onset == 1, "time_sec"]
        onset_m = df.loc[onset == 1, "map"]
        plt.scatter(onset_t, onset_m, label="event onset", marker="x")

        # plot label as band (scaled)
        plt.plot(t, y * 20 + 40, label="y_h5m (scaled)")  # just for visualization

        plt.title(f"Case {caseid}: MAP + event onsets + y_h5m")
        plt.xlabel("time (sec)")
        plt.ylabel("MAP / scaled label")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
