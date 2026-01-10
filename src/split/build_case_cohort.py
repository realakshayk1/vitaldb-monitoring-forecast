from __future__ import annotations

from pathlib import Path
import pandas as pd
import vitaldb
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HR = "Solar8000/HR"
SPO2 = "Solar8000/PLETH_SPO2"
MAP_PRIMARY = "Solar8000/ART_MBP"
MAP_FALLBACK = "Solar8000/NIBP_MBP"

MIN_DURATION_SECONDS = 30 * 60  # 30 minutes

def has_track(caseid: int, track: str) -> bool:
    """
    Fast probe: try loading at 1Hz. If track isn't present, this usually errors.
    Returns True if we get any data back.
    """
    try:
        arr = vitaldb.load_case(caseid, track, interval=1)
        if arr is None:
            return False
        # arr is often a numpy array
        return getattr(arr, "size", 0) > 0 or len(arr) > 0
    except Exception:
        return False

def main(max_cases: int = 200) -> None:
    clinical = pd.read_csv(RAW_DIR / "clinical_data.csv")

    # Duration in seconds (casestart is 0)
    eligible = clinical[clinical["caseend"] >= MIN_DURATION_SECONDS].copy()
    caseids = eligible["caseid"].tolist()

    selected = []
    rows = []

    print(f"Eligible by duration >=30m: {len(caseids)} cases")

    for caseid in tqdm(caseids, desc="Probing cases"):
        if not has_track(caseid, HR):
            continue
        if not has_track(caseid, SPO2):
            continue

        map_source = None
        if has_track(caseid, MAP_PRIMARY):
            map_source = MAP_PRIMARY
        elif has_track(caseid, MAP_FALLBACK):
            map_source = MAP_FALLBACK
        else:
            continue

        selected.append(caseid)
        rows.append({"caseid": caseid, "map_source": map_source})

        if len(selected) >= max_cases:
            break

    (OUT_DIR / "caseids_v1.txt").write_text("\n".join(map(str, selected)) + "\n")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "case_cohort_manifest.csv", index=False)

    print("\n✅ Cohort built")
    print("cases selected:", len(selected))
    print("map source counts:")
    print(manifest["map_source"].value_counts())

if __name__ == "__main__":
    main()
