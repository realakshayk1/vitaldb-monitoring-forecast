from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import lightgbm as lgb

from src.models.split import split_by_case
from src.models.features import split_xy

def main():
    df = pd.read_parquet("data/processed/windows.parquet")
    train, val, test = split_by_case(df)

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_val)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=10)

    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve (validation)")
    plt.show()

if __name__ == "__main__":
    main()
