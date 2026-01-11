from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

def split_by_case(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
):
    """
    Split by caseid so windows from the same case never leak across splits.
    """
    caseids = df["caseid"].unique()
    train_cases, temp_cases = train_test_split(
        caseids, train_size=train_frac, random_state=42
    )
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=0.5, random_state=42
    )

    train = df[df["caseid"].isin(train_cases)]
    val = df[df["caseid"].isin(val_cases)]
    test = df[df["caseid"].isin(test_cases)]

    return train, val, test
