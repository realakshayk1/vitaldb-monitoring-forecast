from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


NN_DIR = Path("reports/nn")
GBM_PATH = Path("reports/gbm/gbm_results.json")

OUT_DIR = Path("reports/compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _add_nn_row(rows: list[dict], path: Path, model_name: str) -> None:
    if not path.exists():
        print(f"⚠️ Missing: {path}")
        return

    obj = _read_json(path)
    agg = obj.get("aggregate", {})

    rows.append({
        "model": model_name,
        "split": "test",
        "auroc_mean": agg.get("test_auroc_mean"),
        "auroc_std": agg.get("test_auroc_std"),
        "auprc_mean": agg.get("test_auprc_mean"),
        "auprc_std": agg.get("test_auprc_std"),
        "source": str(path.as_posix()),
    })


def main():
    rows: list[dict] = []

    # --- Neural nets (CNN/TCN/etc.) ---
    # Prefer explicit files if present
    cnn_path = NN_DIR / "cnn_results.json"
    tcn_path = NN_DIR / "tcn_results.json"

    # If you haven't renamed yet and TCN overwrote cnn_results.json,
    # we can still include it as "NN" by inspecting config.arch.
    if cnn_path.exists():
        obj = _read_json(cnn_path)
        arch = str(obj.get("config", {}).get("arch", "cnn")).lower()
        # If arch says "tcn", label as TCN; else CNN
        label = "TCN" if arch == "tcn" else "CNN"
        _add_nn_row(rows, cnn_path, label)
    else:
        print(f"⚠️ Missing: {cnn_path}")

    # If you later split results correctly, this will add TCN too
    if tcn_path.exists():
        _add_nn_row(rows, tcn_path, "TCN")

    # --- GBM ---
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

    # stable ordering: NN models first, then GBM
    order = {"CNN": 0, "TCN": 1, "GBM": 2}
    df["__order"] = df["model"].map(order).fillna(99)
    df = df.sort_values(["__order", "model"]).drop(columns="__order").reset_index(drop=True)

    out_path = OUT_DIR / "model_auc_comparison.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path}")
    print(df)


if __name__ == "__main__":
    main()
