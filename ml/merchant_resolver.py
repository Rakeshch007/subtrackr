# ml/merchant_resolver.py
from __future__ import annotations
import re
import pandas as pd
from rapidfuzz import fuzz
from .brands import BRAND_RULES

STOPWORDS = {"inc","llc","ltd","co","corp","the","online","payment","purchase",
             "autopay","subscription","renewal","services","service"}

def normalize_merchant(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"http[s]?://\S+"," ", t)
    t = re.sub(r"\d+"," ", t)
    t = re.sub(r"[^a-z\s]"," ", t)
    toks = [w for w in t.split() if w and w not in STOPWORDS]
    return " ".join(toks)[:80].strip()

def soft_group(df: pd.DataFrame, threshold: int = 88) -> pd.DataFrame:
    df = df.copy()
    df["merchant_norm"] = df["description"].apply(normalize_merchant)
    uniq = df["merchant_norm"].dropna().unique().tolist()
    mapping = {}
    for i, u in enumerate(uniq):
        if not u or u in mapping: 
            continue
        mapping[u] = u
        for v in uniq[i+1:]:
            if not v or v in mapping: 
                continue
            if fuzz.token_set_ratio(u, v) >= threshold:
                mapping[v] = u
    df["merchant_norm"] = df["merchant_norm"].map(lambda x: mapping.get(x, x))
    return df

def apply_brand_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    brands, cats = [], []
    for desc in df["description"].astype(str):
        hit_b, hit_c = None, None
        for rule in BRAND_RULES:
            if rule.pattern.search(desc):
                hit_b, hit_c = rule.name, rule.category
                break
        brands.append(hit_b); cats.append(hit_c)
    df["brand"] = brands
    df["category"] = cats
    df["brand_hit"] = df["brand"].notna().astype(int)
    return df

def resolve_merchants(tx: pd.DataFrame) -> pd.DataFrame:
    tx = soft_group(tx)
    tx = apply_brand_lexicon(tx)
    return tx
