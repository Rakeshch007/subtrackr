# ml/weak_labels.py
from __future__ import annotations
import re
import pandas as pd

NEGATIVE_MERCHANTS = [
    "gas","fuel","station","walmart","target","mcdonald","grocery","store","supermarket",
    "liquor","shell","chevron","7-eleven","7eleven","costco","aldi","kroger","tesco","carrefour","spar"
]
NEG_TOKENS = re.compile("|".join(map(re.escape, NEGATIVE_MERCHANTS)), re.I)

def weak_label(feat: pd.DataFrame) -> pd.Series:
    """
    High-precision weak labels for training:
      • require positive evidence (brand_hit OR hint_flag)
      • yearly: count>=1 & span>=300
      • weekly/biweekly/monthly: count>=2 & cv<=0.35
      • exclude obvious retailish names
      • reasonable amount
    """
    reasonable_amt = feat["mean_amt"].between(1.0, 300.0)
    positive = (feat["brand_hit"]==1) | (feat["hint_flag"]==1)

    yearly = (feat["is_yearly"]==1) & (feat["span_days"]>=300) & (feat["count"]>=1)
    cyclic = (
        ((feat["is_monthly"]==1) | (feat["is_biweekly"]==1) | (feat["is_weekly"]==1))
        & (feat["count"]>=2) & (feat["cv"]<=0.35)
    )

    mnorm = feat["merchant_norm"].fillna("")
    not_retailish = ~mnorm.str.contains(NEG_TOKENS)

    label = (reasonable_amt & (yearly | cyclic) & positive & not_retailish)
    return label.astype(int)
