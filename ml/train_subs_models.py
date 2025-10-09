# ml/train_subs_models.py
from __future__ import annotations
import os, json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from .features import FEATURES, build_feature_table
from .weak_labels import weak_label
from .merchant_resolver import resolve_merchants

META_VERSION = "subs_v1"

def _prepare_xy(tx: pd.DataFrame):
    tx2  = resolve_merchants(tx)
    feat = build_feature_table(tx2)
    y    = weak_label(feat).values
    X    = feat[FEATURES].values
    return X, y, feat

def train_from_transactions(tx: pd.DataFrame, out_dir="models") -> None:
    """
    Train RandomForest + XGBoost on weak labels and save artifacts:
      - rf_subscription.pkl
      - xgb_subscription.pkl
      - subs_meta.json (feature list, version)
    """
    X, y, feat = _prepare_xy(tx)
    if len(np.unique(y)) < 2:
        raise ValueError("Training labels contain a single class; provide more diverse data.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # --- RandomForest with probability calibration ---
    rf_base = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf = CalibratedClassifierCV(rf_base, method="isotonic", cv=3)
    rf.fit(Xtr, ytr)

    # --- XGBoost with imbalance weight ---
    pos = (ytr==1).sum(); neg = (ytr==0).sum()
    scale_pos_weight = (neg/pos) if pos > 0 else 1.0
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
    xgb.fit(Xtr, ytr)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf,  os.path.join(out_dir, "rf_subscription.pkl"))
    joblib.dump(xgb, os.path.join(out_dir, "xgb_subscription.pkl"))

    meta = {
        "version": META_VERSION,
        "features": FEATURES,
        "n_samples": int(len(y)),
        "class_balance": {"pos": int((y==1).sum()), "neg": int((y==0).sum())}
    }
    with open(os.path.join(out_dir, "subs_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
