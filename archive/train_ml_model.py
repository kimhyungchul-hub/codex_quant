from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss


def build_features(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    closes = df[close_col].astype(float).values
    feats = {}
    arr = np.asarray(closes, dtype=float)
    for lag in (1, 5, 20):
        feats[f"ret_{lag}"] = np.concatenate([[0.0] * lag, arr[lag:] / arr[:-lag] - 1.0])
    for win in (5, 20, 60):
        rets = np.diff(np.log(arr))
        vol = np.concatenate([[0.0] * (win + 1), pd.Series(rets).rolling(win).std().fillna(0.0).values])
        feats[f"vol_{win}"] = vol
    # simple z-score
    z20 = (arr - pd.Series(arr).rolling(20).mean()) / (pd.Series(arr).rolling(20).std() + 1e-9)
    feats["z20"] = z20.fillna(0.0).values
    X = pd.DataFrame(feats)
    return X


def build_target(df: pd.DataFrame, close_col: str = "close") -> np.ndarray:
    arr = df[close_col].astype(float).values
    fwd = np.concatenate([arr[1:], arr[-1:]])
    ret_fwd = fwd / arr - 1.0
    y = (ret_fwd > 0).astype(int)
    return y


def train_model(csv_path: Path, close_col: str, out_path: Path):
    df = pd.read_csv(csv_path)
    X = build_features(df, close_col=close_col)
    y = build_target(df, close_col=close_col)

    # time series split
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    briers = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        base = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=3)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
        briers.append(brier_score_loss(y_test, proba))

    # final fit on full data
    base = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=3)
    clf.fit(X, y)

    obj = {"model": clf, "features": list(X.columns), "metrics": {"auc_cv": float(np.mean(aucs)), "brier_cv": float(np.mean(briers))}}
    joblib.dump(obj, out_path)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Train calibrated ML model for next-period direction.")
    parser.add_argument("--csv", required=True, help="CSV file path")
    parser.add_argument("--col", default="close", help="Close column name")
    parser.add_argument("--out", default="ml_model.pkl", help="Output model path")
    args = parser.parse_args()
    obj = train_model(Path(args.csv), args.col, Path(args.out))
    print("Saved model to", args.out)
    print("CV metrics:", json.dumps(obj["metrics"], indent=2))


if __name__ == "__main__":
    main()

