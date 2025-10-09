# ml/recurring.py
from __future__ import annotations
import pandas as pd, numpy as np

CADENCES = [("weekly",7,1), ("biweekly",14,2), ("monthly",30,3), ("quarterly",90,7), ("yearly",365,10)]
HINTS = {"subscription","subs","member","membership","premium","plus","plan","auto pay","autopay","renewal"}

def _cadence_label(med_gap: float) -> str | None:
    for label, base, wiggle in CADENCES:
        if abs(med_gap - base) <= wiggle or (label=="monthly" and 28<=med_gap<=31):
            return label
    return None

def detect_recurring_subscriptions(tx: pd.DataFrame, min_occurrences=3, max_cv=0.25) -> pd.DataFrame:
    if tx.empty:
        return pd.DataFrame(columns=["merchant_norm","brand","category","count","mean_amt","cv","cadence","last_date","next_expected","is_recurring","is_subscription"])
    df = tx.copy(); df["date"] = pd.to_datetime(df["date"])

    rows = []
    for m, g in df.groupby("merchant_norm"):
        g = g.sort_values("date")
        dates = pd.to_datetime(g["date"])
        gaps = dates.diff().dt.days.dropna().to_numpy()
        med_gap = float(np.median(gaps)) if gaps.size else 0.0
        cadence = _cadence_label(med_gap)

        abs_amt = g["amount"].abs().astype(float)
        mean_amt = float(abs_amt.mean()) if len(abs_amt) else 0.0
        cv = float(abs_amt.std(ddof=0) / (mean_amt + 1e-9)) if mean_amt > 0 else 999.0
        count = int(len(g))

        has_min = count >= min_occurrences
        stable = cv <= max_cv
        cadence_ok = cadence in {"weekly","biweekly","monthly","yearly"}
        is_recurring = bool(has_min and stable and cadence_ok)

        desc_blob = " ".join(g["description"].astype(str)).lower()
        hint = any(w in desc_blob for w in HINTS)
        brand_hit = int(g["brand"].notna().any())
        category = (g["category"].dropna().iloc[0]) if g["category"].notna().any() else None

        in_amount_band = 4.0 <= mean_amt <= 250.0
        is_subscription = bool(is_recurring and (brand_hit or hint or (in_amount_band and cadence=="monthly")))

        last_date = dates.max().date() if len(dates) else None
        days = 30
        if cadence == "weekly": days = 7
        elif cadence == "biweekly": days = 14
        elif cadence == "quarterly": days = 90
        elif cadence == "yearly": days = 365
        next_expected = (pd.to_datetime(last_date) + pd.Timedelta(days=days)).date() if last_date else None

        rows.append({
            "merchant_norm": m, "brand": (g["brand"].dropna().iloc[0]) if brand_hit else None,
            "category": category, "count": count, "mean_amt": round(mean_amt,2),
            "cv": round(cv,3), "cadence": cadence, "last_date": last_date,
            "next_expected": next_expected, "is_recurring": is_recurring, "is_subscription": is_subscription
        })
    return pd.DataFrame(rows).sort_values(["is_subscription","count","mean_amt"], ascending=[False,False,False]).reset_index(drop=True)
