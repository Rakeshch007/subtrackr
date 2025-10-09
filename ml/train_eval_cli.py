# ml/train_eval_cli.py
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

from .parse_transactions import parse_text_to_transactions
from .merchant_resolver import resolve_merchants
from .features import FEATURES, build_feature_table
from .weak_labels import weak_label

def load_texts(paths: list[str]) -> str:
    """Concatenate many .txt files into one big string."""
    chunks = []
    for p in paths:
        chunks.append(Path(p).read_text(encoding="utf-8"))
    return "\n".join(chunks)

def prepare_xy(tx: pd.DataFrame):
    tx2 = resolve_merchants(tx)
    feat = build_feature_table(tx2)
    y = weak_label(feat).values
    X = feat[FEATURES].values
    return X, y, feat

def train_models(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # RF + calibration (for good probabilities)
    rf_base = RandomForestClassifier(
        n_estimators=600, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf = CalibratedClassifierCV(rf_base, method="isotonic", cv=3)
    rf.fit(Xtr, ytr)

    # XGB with imbalance weight
    pos, neg = (ytr == 1).sum(), (ytr == 0).sum()
    spw = (neg / pos) if pos else 1.0
    xgb = XGBClassifier(
        n_estimators=800, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, eval_metric="logloss",
        scale_pos_weight=spw
    )
    xgb.fit(Xtr, ytr)

    return (rf, xgb, Xtr, Xte, ytr, yte)

def evaluate(name, model, Xtr, Xte, ytr, yte, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p_tr = model.predict_proba(Xtr)[:,1]
    p_te = model.predict_proba(Xte)[:,1]

    auc_tr = roc_auc_score(ytr, p_tr) if len(np.unique(ytr))>1 else float("nan")
    auc_te = roc_auc_score(yte, p_te) if len(np.unique(yte))>1 else float("nan")

    yhat = (p_te >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary", zero_division=0)
    cm = confusion_matrix(yte, yhat)

    report = {
        "model": name,
        "AUC_train": float(auc_tr),
        "AUC_test": float(auc_te),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(yte)),
        "pos_rate_test": float(yte.mean())
    }
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== {name} ===")
    print(json.dumps(report, indent=2))

    # Quick plots (optional)
    try:
        RocCurveDisplay.from_predictions(yte, p_te)
        plt.title(f"ROC — {name}")
        plt.savefig(os.path.join(out_dir, f"{name}_roc.png"), dpi=160, bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(yte, p_te)
        plt.title(f"PR — {name}")
        plt.savefig(os.path.join(out_dir, f"{name}_pr.png"), dpi=160, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser(description="Train & evaluate RF and XGB for subscription detection.")
    ap.add_argument("--txt-glob", default="synthetic_data/*.txt", help="Glob for training TXT files.")
    ap.add_argument("--out-dir", default="models", help="Where to save models & metrics.")
    ap.add_argument("--save", action="store_true", help="Save trained models to --out-dir.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.txt_glob))
    if not paths:
        raise SystemExit(f"No training .txt found for pattern: {args.txt_glob}\nRun your generator first.")
    raw = load_texts(paths)
    tx = parse_text_to_transactions(raw)
    if tx.empty:
        raise SystemExit("Parser produced 0 transactions — check your input TXT format.")
    X, y, feat = prepare_xy(tx)
    if len(np.unique(y)) < 2:
        raise SystemExit("Weak labels produced a single class — add more months/variety.")

    rf, xgb, Xtr, Xte, ytr, yte = train_models(X, y)

    # Evaluate
    evaluate("rf_subscription", rf, Xtr, Xte, ytr, yte, args.out_dir)
    evaluate("xgb_subscription", xgb, Xtr, Xte, ytr, yte, args.out_dir)

    # Save models (and a tiny meta file)
    if args.save:
        joblib.dump(rf,  os.path.join(args.out_dir, "rf_subscription.pkl"))
        joblib.dump(xgb, os.path.join(args.out_dir, "xgb_subscription.pkl"))
        with open(os.path.join(args.out_dir, "subs_meta.json"), "w") as f:
            json.dump({"features": FEATURES, "version": "subs_v1"}, f)
        print(f"\n[ok] Saved models → {args.out_dir}/rf_subscription.pkl and xgb_subscription.pkl")

if __name__ == "__main__":
    main()
