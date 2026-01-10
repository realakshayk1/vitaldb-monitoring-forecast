from __future__ import annotations

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")


def main() -> None:
    clinical_path = RAW_DIR / "clinical_data.csv"
    tracks_path = RAW_DIR / "track_names.csv"

    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing: {clinical_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing: {tracks_path}")

    clinical = pd.read_csv(clinical_path)
    tracks = pd.read_csv(tracks_path)

    print("✅ Loaded:")
    print(" clinical_data:", clinical.shape)
    print(" track_names:", tracks.shape)
    print()

    # Show columns so we know what we're working with
    print("clinical_data columns:")
    print(list(clinical.columns))
    print()

    print("track_names columns:")
    print(list(tracks.columns))
    print()

    # --- Step 2.5 patch: VitalDB track dictionary uses `Parameter` ---
    track_col = "Parameter"
    if track_col not in tracks.columns:
        raise KeyError(
            f"Expected '{track_col}' column in track_names.csv, "
            f"but found: {list(tracks.columns)}"
        )

    all_tnames = tracks[track_col].astype(str)

    # Track strings we care about (can adjust after we inspect exact naming)
    candidates = [
        "Solar8000/HR",
        "Solar8000/SPO2",
        "Solar8000/ART_MBP",
        "Solar8000/NIBP_MBP",
        "SNUADC/ART",
        "SNUADC/PLETH",
    ]

    print("Candidate track presence (exact string match):")
    for t in candidates:
        print(f"  {t:20s} ->", (all_tnames == t).sum())

    # --- Step 2.5: explicit SPO2 confirmation ---
    print("\nDoes SPO2 exist?")
    print((all_tnames == "Solar8000/SPO2").sum())

    print("\nSPO2-like tracks:")
    spo2_like = all_tnames[all_tnames.str.contains("SPO2", case=False, na=False)]
    if len(spo2_like) == 0:
        print("(none found)")
    else:
        print(spo2_like.drop_duplicates().to_string(index=False))

    # Helpful: search for approximate matches (e.g., ART_MBP vs ART_Mean)
    print("\nApprox matches for MAP-like tracks:")
    mask_map = all_tnames.str.contains("MBP|MAP|Mean", case=False, na=False)
    map_like = all_tnames[mask_map].drop_duplicates()
    if len(map_like) == 0:
        print("(none found)")
    else:
        print(map_like.head(50).to_string(index=False))

    print("\nApprox matches for HR-like tracks:")
    mask_hr = all_tnames.str.contains(r"\bHR\b|Heart", case=False, na=False, regex=True)
    hr_like = all_tnames[mask_hr].drop_duplicates()
    if len(hr_like) == 0:
        print("(none found)")
    else:
        print(hr_like.head(50).to_string(index=False))


if __name__ == "__main__":
    main()

"""Inspect VitalDB metadata to identify relevant track names."""
