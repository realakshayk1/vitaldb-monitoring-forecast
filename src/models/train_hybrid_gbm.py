from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = Path("data/processed/windows.parquet")
EMB_DIR = Path("data/processed/embeddings")
SPLIT_JSON = Path("data/processed/seq/norm_stats.json")

OUT_DIR = Path("reports/hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Config knobs (edit these)
# -----------------------------
ARCH = "cnn"                 # "cnn" or "tcn" (if you extracted tcn embeddings)
TOL_SEC = 0                  # set to 5 if you suspect time anchor mismatch
FEATURE_MODE = "emb_only"
# options:
#   "emb_only"
#   "tabular_plus_emb"  (uses more windows.parquet features)


# -----------------------------
# IO helpers
# -----------------------------
def load_case_splits() -> dict:
    obj = json.loads(SPLIT_JSON.read_text())
    return {
        "train": set(obj["train_cases"]),
        "val": set(obj["val_cases"]),
        "test": set(obj["test_cases"]),
    }


def load_embeddings(arch: str) -> pd.DataFrame:
    parts = []
    for split in ["train", "val", "test"]:
        p = EMB_DIR / f"{arch}_{split}_embeddings.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing embeddings: {p}. Run python -m src.nn.extract_embeddings first."
            )
        df = pd.read_parquet(p)
        df["split"] = split
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)

    out["caseid"] = out["caseid"].astype(int)
    out["time_sec"] = out["time_sec"].astype(int)
    return out


def merge_with_tolerance(emb: pd.DataFrame, win: pd.DataFrame, tol_sec: int = 5) -> pd.DataFrame:
    """
    Merge each case on nearest time within +/- tol_sec seconds.
    Use when you suspect time anchoring mismatch (start vs end).
    """
    emb = emb.sort_values(["caseid", "time_sec"]).reset_index(drop=True)
    win = win.sort_values(["caseid", "time_sec"]).reset_index(drop=True)

    out_parts = []
    for caseid, e in emb.groupby("caseid", sort=False):
        w = win[win["caseid"] == caseid]
        if w.empty:
            continue

        m = pd.merge_asof(
            e.sort_values("time_sec"),
            w.sort_values("time_sec"),
            on="time_sec",
            direction="nearest",
            tolerance=tol_sec,
        )
        m["caseid"] = int(caseid)
        out_parts.append(m)

    return pd.concat(out_parts, ignore_index=True) if out_parts else emb.iloc[0:0]


# -----------------------------
# Feature selection
# -----------------------------
def pick_feature_cols(merged: pd.DataFrame, feature_mode: str) -> list[str]:
    """
    Goal: prevent the GBM from just re-learning the same instantaneous vitals the CNN saw.
    - emb_only: only embedding columns
    - static_plus_emb: embedding + slow/static (age/sex/bmi/asa/etc.)
    - tabular_plus_emb: embedding + broader windows.parquet tabular set (still excludes IDs/labels)
    """
    ycol = "y_h5m"
    drop_cols = {ycol, "caseid", "time_sec", "split"}
    if "y" in merged.columns:
        drop_cols.add("y")

    # heuristic: treat embedding columns as those that start with "emb_"
    emb_cols = [c for c in merged.columns if c.startswith("emb_")]

    # If your extractor didn’t prefix, fallback to numeric-only excluding known cols
    if not emb_cols:
        # common alternative: "z0..z63" or "feat_0.."
        emb_cols = [c for c in merged.columns if c.startswith(("z", "feat_", "latent_"))]

    static_candidates = [
        "age", "sex", "height", "weight", "bmi", "asa", "emop",
        # include any other "slow" features you know exist
    ]
    static_cols = [c for c in static_candidates if c in merged.columns]

    # tabular "all" features (minus leakage/id/labels)
    tabular_cols = [c for c in merged.columns if c not in drop_cols and c not in emb_cols]

    if feature_mode == "emb_only":
        feats = emb_cols
    elif feature_mode == "static_plus_emb":
        feats = emb_cols + static_cols
    elif feature_mode == "tabular_plus_emb":
        feats = emb_cols + tabular_cols
    else:
        raise ValueError(f"Unknown feature_mode={feature_mode}")

    # remove any accidental drops
    feats = [c for c in feats if c not in drop_cols]

    if len(feats) == 0:
        raise RuntimeError(
            f"No features selected. emb_cols={len(emb_cols)} static_cols={len(static_cols)} "
            f"tabular_cols={len(tabular_cols)}"
        )

    return feats


# -----------------------------
# Eval
# -----------------------------
def eval_metrics(model: lgb.Booster, X: pd.DataFrame, y: pd.Series) -> dict:
    p = model.predict(X)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def main():
    # Load
    df = pd.read_parquet(DATA_PATH)
    emb = load_embeddings(ARCH)

    df["caseid"] = df["caseid"].astype(int)
    df["time_sec"] = df["time_sec"].astype(int)

    # Diagnostics
    print("windows:", len(df), "embeddings:", len(emb))
    print("windows time_sec range:", int(df["time_sec"].min()), int(df["time_sec"].max()))
    print("emb time_sec range:", int(emb["time_sec"].min()), int(emb["time_sec"].max()))

    # Merge direction: embeddings are the master index
    if TOL_SEC <= 0:
        merged = emb.merge(df, on=["caseid", "time_sec"], how="left", suffixes=("", "_win"))
        missing_rate = float(merged["y_h5m"].isna().mean())
        print(f"after left-join, missing tabular rows: {missing_rate:.2%}")
    else:
        merged = merge_with_tolerance(emb, df, tol_sec=TOL_SEC)
        missing_rate = float(merged["y_h5m"].isna().mean())
        print(f"after tolerance-join (±{TOL_SEC}s), missing tabular rows: {missing_rate:.2%}")

    if missing_rate > 0.05:
        print("⚠️ Large missing rate after join. Hybrid results may be untrustworthy.")
        print("   Try TOL_SEC=5 or 10. If still high, rebuild windows on the NN time grid.")

    # Label agreement (optional)
    if "y" in merged.columns:
        mask = merged["y_h5m"].notna()
        disagree = (merged.loc[mask, "y_h5m"].astype(int) != merged.loc[mask, "y"].astype(int)).mean()
        print(f"label_disagreement_rate={float(disagree):.6f}")

    # Splits
    splits = load_case_splits()
    train = merged[merged["caseid"].isin(splits["train"])].copy()
    val   = merged[merged["caseid"].isin(splits["val"])].copy()
    test  = merged[merged["caseid"].isin(splits["test"])].copy()

    # drop rows with missing labels
    ycol = "y_h5m"
    train = train[train[ycol].notna()]
    val   = val[val[ycol].notna()]
    test  = test[test[ycol].notna()]

    # Feature set
    feature_cols = pick_feature_cols(merged, FEATURE_MODE)
    print(f"FEATURE_MODE={FEATURE_MODE} | n_features={len(feature_cols)}")
    print("feature sample:", feature_cols[:20])

    X_train, y_train = train[feature_cols], train[ycol].astype(int)
    X_val, y_val     = val[feature_cols],   val[ycol].astype(int)
    X_test, y_test   = test[feature_cols],  test[ycol].astype(int)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds   = lgb.Dataset(X_val, label=y_val)

    # ✅ Reduced-capacity GBM (less overfit on embeddings)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=4000,
        callbacks=[lgb.early_stopping(200)],
    )

    val_metrics = eval_metrics(model, X_val, y_val)
    test_metrics = eval_metrics(model, X_test, y_test)

    print(f"[HYBRID-{ARCH} | {FEATURE_MODE}] val:  AUROC={val_metrics['auroc']:.3f} | AUPRC={val_metrics['auprc']:.3f}")
    print(f"[HYBRID-{ARCH} | {FEATURE_MODE}] test: AUROC={test_metrics['auroc']:.3f} | AUPRC={test_metrics['auprc']:.3f}")

    # Save
    model_path = MODEL_DIR / f"hybrid_{ARCH}_{FEATURE_MODE}.txt"
    model.save_model(str(model_path))

    out = {
        "arch": ARCH,
        "feature_mode": FEATURE_MODE,
        "tol_sec": int(TOL_SEC),
        "rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "missing_rate_after_join": float(missing_rate),
        "n_features": int(len(feature_cols)),
        "val": val_metrics,
        "test": test_metrics,
        "feature_cols_sample": feature_cols[:50],
        "lgb_params": params,
        "best_iteration": int(model.best_iteration),
    }
    out_path = OUT_DIR / f"hybrid_{ARCH}_{FEATURE_MODE}_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"✅ Saved model: {model_path}")
    print(f"✅ Saved results: {out_path}")


if __name__ == "__main__":
    main()
