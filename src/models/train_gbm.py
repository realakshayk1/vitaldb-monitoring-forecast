from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.split import split_by_case
from src.models.features import split_xy

DATA_PATH = "data/processed/windows.parquet"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("reports/gbm")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _metrics(y, p) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def main():
    df = pd.read_parquet(DATA_PATH)
    train, val, test = split_by_case(df)

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50)],
    )

    # metrics
    val_p = model.predict(X_val)
    test_p = model.predict(X_test)

    val_m = _metrics(y_val, val_p)
    test_m = _metrics(y_test, test_p)

    print(f"[GBM] val:  AUROC={val_m['auroc']:.3f} | AUPRC={val_m['auprc']:.3f}")
    print(f"[GBM] test: AUROC={test_m['auroc']:.3f} | AUPRC={test_m['auprc']:.3f}")

    # save model
    model_path = MODEL_DIR / "gbm.txt"
    model.save_model(str(model_path))

    # save report
    payload = {
        "model": "GBM",
        "data_path": DATA_PATH,
        "params": params,
        "best_iteration": int(model.best_iteration) if model.best_iteration else None,
        "val": val_m,
        "test": test_m,
        "model_path": str(model_path.as_posix()),
    }

    out_path = REPORT_DIR / "gbm_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"📄 Saved: {out_path}")


if __name__ == "__main__":
    main()
