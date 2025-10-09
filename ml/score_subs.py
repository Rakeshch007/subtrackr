# ml/score_subs.py
from __future__ import annotations
import os, json, re
import joblib
import pandas as pd
from .features import FEATURES, build_feature_table
from .merchant_resolver import resolve_merchants

# Common recurring but non-subscription merchants/terms.
# These are only blocked when there is no positive evidence (brand/hint).
BLOCK_WORDS = [
    "gas", "fuel", "station", "walmart", "target", "mcdonald",
    "grocery", "store", "supermarket", "liquor", "shell",
    "chevron", "7-eleven", "7eleven", "costco", "aldi",
    "kroger", "tesco", "carrefour", "spar"
]
BLOCK_PATTERN = re.compile("|".join(map(re.escape, BLOCK_WORDS)), re.I)


class SubscriptionScorer:
    """
    Loads a trained model and scores merchants.
    - Reads subs_meta.json (if provided) to align feature order/shape.
    - Uses a configurable threshold (default 0.65) for higher precision.
    - Applies a safety post-filter to cut retail-like false positives
      unless there is positive evidence (brand/hint).
    - Fixes merge bug so 'count' and 'mean_amt' populate correctly.
    """

    def __init__(self, model_path: str, meta_path: str | None = None, threshold: float = 0.65):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        self.threshold = float(threshold)

        # Determine feature list used by the trained model
        self.meta_features = None
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                feats = meta.get("features")
                if isinstance(feats, list) and feats:
                    self.meta_features = feats
            except Exception:
                pass
        if self.meta_features is None:
            # Fallback to runtime FEATURES (only safe if model was trained with same list)
            self.meta_features = FEATURES

    def _align_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        # Add any missing features as zeros; drop extras; order exactly as in training
        for col in self.meta_features:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        return feat_df[self.meta_features].copy()

    def score(self, tx: pd.DataFrame) -> pd.DataFrame:
        # 1) Normalize merchants + build features
        tx2 = resolve_merchants(tx)
        feat = build_feature_table(tx2)

        # 2) Predict probabilities with aligned feature matrix
        X = self._align_features(feat)
        probs = self.model.predict_proba(X.values)[:, 1]
        feat = feat.assign(prob=probs, is_subscription=(probs >= self.threshold).astype(int))

        # 3) Attach group stats for display (count/mean_amt/brand/category)
        add = (
            tx2.groupby("merchant_norm")
               .agg(
                   brand=("brand", "first"),
                   category=("category", "first"),
                   count=("merchant_norm", "size"),
                   mean_amt=("amount", lambda s: float(abs(s).mean())),
               )
               .reset_index()
        )

        # Keep evidence columns for post-filter; drop dup numeric cols before merge
        keep_cols = ["merchant_norm", "prob", "is_subscription", "brand_hit", "hint_flag", "neg_name_flag"]
        feat_for_merge = feat[keep_cols]
        out = feat_for_merge.merge(add, on="merchant_norm", how="left")

        # ---- Safety post-filter: block common retail-like names w/o positive evidence ----
        block_retail = out["merchant_norm"].fillna("").str.contains(BLOCK_PATTERN)
        has_positive = (out["brand"].notna()) | (out["brand_hit"] == 1) | (out["hint_flag"] == 1)
        mask = block_retail & (~has_positive)

        # Down-weight probability and force negative when blocked
        out.loc[mask, "prob"] = out.loc[mask, "prob"] * 0.2
        out.loc[mask, "is_subscription"] = 0

        # 4) Ensure columns exist and are usable in UI
        out["count"] = out["count"].fillna(0).astype(int)
        out["mean_amt"] = out["mean_amt"].fillna(0.0)
        if "brand" not in out.columns:
            out["brand"] = None
        if "category" not in out.columns:
            out["category"] = None

        cols = ["merchant_norm", "brand", "category", "prob", "is_subscription", "count", "mean_amt"]
        return out[cols].sort_values("prob", ascending=False)
