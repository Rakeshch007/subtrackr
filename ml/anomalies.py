# ml/anomalies.py
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

def flag_amount_anomalies(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Per-merchant, flag amount outliers (spikes/drops). Uses IsolationForest
    when there are enough points; otherwise falls back to simple z-score.
    """
    parts = []
    for m, g in tx.groupby("merchant_norm"):
        gg = g.copy()
        X = gg["amount"].abs().to_numpy().reshape(-1,1)
        if len(gg) >= 6:
            iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
            pred = iso.fit_predict(X)  # -1 = anomaly
            gg["amount_anomaly"] = (pred == -1).astype(int)
        else:
            mu = X.mean(); sd = X.std() if X.std() > 0 else 1.0
            z = (X - mu) / sd
            gg["amount_anomaly"] = (np.abs(z) >= 3).astype(int)
        parts.append(gg)
    return pd.concat(parts, ignore_index=True) if parts else tx

def flag_missed_cycles(subs: pd.DataFrame) -> pd.DataFrame:
    """
    If a subscription is overdue by >1.5× expected cadence, mark missed_cycle=1.
    """
    if subs.empty:
        subs["missed_cycle"] = []
        return subs
    subs = subs.copy()
    map_days = {"weekly":7, "biweekly":14, "monthly":30, "quarterly":90, "yearly":365}
    def missed(row):
        if not row["is_subscription"] or not row["last_date"] or not row["cadence"]:
            return 0
        d = pd.to_datetime(row["last_date"])
        days = map_days.get(row["cadence"], 30)
        return int(pd.Timestamp.today().normalize() > d + pd.Timedelta(days=int(1.5*days)))
    subs["missed_cycle"] = subs.apply(missed, axis=1)
    return subs
