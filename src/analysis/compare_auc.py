from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CNN_PATH = Path("reports/nn/cnn_results.json")
GBM_PATH = Path("reports/gbm/gbm_results.json")

OUT_DIR = Path("reports/compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def main():
    rows = []

    if CNN_PATH.exists():
        cnn = _read_json(CNN_PATH)
        agg = cnn.get("aggregate", {})
        rows.append({
            "model": "CNN",
            "split": "test",
            "auroc_mean": agg.get("test_auroc_mean"),
            "auroc_std": agg.get("test_auroc_std"),
            "auprc_mean": agg.get("test_auprc_mean"),
            "auprc_std": agg.get("test_auprc_std"),
            "source": str(CNN_PATH.as_posix()),
        })
    else:
        print(f"⚠️ Missing: {CNN_PATH}")

    if GBM_PATH.exists():
        gbm = _read_json(GBM_PATH)
        rows.append({
            "model": "GBM",
            "split": "test",
            "auroc_mean": gbm.get("test", {}).get("auroc"),
            "auroc_std": None,
            "auprc_mean": gbm.get("test", {}).get("auprc"),
            "auprc_std": None,
            "source": str(GBM_PATH.as_posix()),
        })
    else:
        print(f"⚠️ Missing: {GBM_PATH}")

    if not rows:
        raise RuntimeError("No model result files found to compare.")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "model_auc_comparison.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path}")
    print(df)


if __name__ == "__main__":
    main()
