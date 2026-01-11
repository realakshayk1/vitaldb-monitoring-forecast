from __future__ import annotations

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.split import split_by_case
from src.models.features import split_xy

DATA_PATH = "data/processed/windows.parquet"

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

    for name, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        p = model.predict(X)
        print(
            f"[GBM] {name}: "
            f"AUROC={roc_auc_score(y, p):.3f} | "
            f"AUPRC={average_precision_score(y, p):.3f}"
        )
    model.save_model("models/gbm.txt")

if __name__ == "__main__":
    main()
