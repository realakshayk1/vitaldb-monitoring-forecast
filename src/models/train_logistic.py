from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ))
    ])

    pipe.fit(X_train, y_train)

    for name, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        p = pipe.predict_proba(X)[:, 1]
        print(
            f"[LogReg] {name}: "
            f"AUROC={roc_auc_score(y, p):.3f} | "
            f"AUPRC={average_precision_score(y, p):.3f}"
        )

if __name__ == "__main__":
    main()
