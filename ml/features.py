# ml/features.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict

HINTS = {"subscription","subs","member","membership","premium","plus","plan","autopay","auto pay","renewal"}

# words that commonly indicate *non-subscription* recurring spend
NEG_WORDS = {
    "gas","fuel","station","walmart","target","mcdonald","grocery","store","supermarket",
    "liquor","shell","chevron","7-eleven","7eleven","costco","aldi","kroger","tesco","carrefour","spar"
}

CADENCES = [("weekly",7,1), ("biweekly",14,2), ("monthly",30,3), ("quarterly",90,7), ("yearly",365,15)]

def _cadence_label(med_gap: float) -> str | None:
    for label, base, wiggle in CADENCES:
        if abs(med_gap - base) <= wiggle or (label=="monthly" and 28<=med_gap<=31):
            return label
    return None

def _cadence_flags(label: str | None) -> Dict[str,int]:
    return {
        "is_weekly": int(label=="weekly"),
        "is_biweekly": int(label=="biweekly"),
        "is_monthly": int(label=="monthly"),
        "is_quarterly": int(label=="quarterly"),
        "is_yearly": int(label=="yearly"),
    }

def group_features(group: pd.DataFrame) -> dict:
    g = group.copy()
    g["date"] = pd.to_datetime(g["date"])
    dates = g["date"].sort_values()
    gaps = dates.diff().dt.days.dropna().to_numpy()
    med_gap = float(np.median(gaps)) if gaps.size else 0.0
    gap_std = float(np.std(gaps)) if gaps.size else 0.0
    cad = _cadence_label(med_gap)

    amounts = g["amount"].astype(float)
    abs_amt  = amounts.abs()
    mean_amt = float(abs_amt.mean()) if len(abs_amt) else 0.0
    std_amt  = float(abs_amt.std(ddof=0)) if len(abs_amt) else 0.0
    cv       = (std_amt / (mean_amt + 1e-9)) if mean_amt > 0 else 999.0

    count    = int(len(g))
    span_days= int((dates.max()-dates.min()).days) if len(dates)>1 else 0
    debit_ratio = float((amounts < 0).mean()) if len(amounts) else 0.0

    desc_blob = " ".join(g["description"].astype(str)).lower()
    hint_flag = int(any(w in desc_blob for w in HINTS))
    brand_hit = int(g["brand"].notna().any())

    merchant = str(g["merchant_norm"].iloc[0] or "").lower()
    neg_name_flag = int(any(w in merchant for w in NEG_WORDS))

    flags = _cadence_flags(cad)
    return {
        "merchant_norm": g["merchant_norm"].iloc[0],
        "brand_hit": brand_hit,
        "hint_flag": hint_flag,
        "neg_name_flag": neg_name_flag,
        "count": count,
        "span_days": span_days,
        "med_gap": med_gap,
        "gap_std": gap_std,
        "mean_amt": mean_amt,
        "cv": cv,
        "debit_ratio": debit_ratio,
        **flags,
    }

FEATURES = [
    "brand_hit","hint_flag","neg_name_flag",
    "count","span_days","med_gap","gap_std",
    "mean_amt","cv","debit_ratio",
    "is_weekly","is_biweekly","is_monthly","is_quarterly","is_yearly",
]

def build_feature_table(tx: pd.DataFrame) -> pd.DataFrame:
    feats = [group_features(g) for _, g in tx.groupby("merchant_norm")]
    return pd.DataFrame(feats)
