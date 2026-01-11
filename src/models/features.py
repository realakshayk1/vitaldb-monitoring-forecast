from __future__ import annotations
import pandas as pd

EXCLUDE_COLS = {"caseid", "time_sec", "y_h5m"}

def split_xy(df: pd.DataFrame):
    X = df.drop(columns=EXCLUDE_COLS)
    y = df["y_h5m"].astype(int)
    return X, y
