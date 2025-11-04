#!/usr/bin/env python
"""
Export a merchant-level CSV for human labeling.

Input:
  --txt-glob "synthetic_data/*.txt"  (or any TXT glob your parser supports)

Output:
  data/merchants_for_labeling.csv with columns:
    merchant_norm, sample_descriptions, brand, category,
    count, span_days, mean_amt, cv, is_weekly, is_biweekly, is_monthly, is_quarterly, is_yearly,
    human_label   <-- FILL THIS (1 = subscription, 0 = not subscription)
"""

from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import pandas as pd

# local imports (project structure)
from ml.parse_transactions import parse_text_to_transactions
from ml.merchant_resolver import resolve_merchants
from ml.features import build_feature_table

def load_texts(paths: list[str]) -> str:
    chunks = []
    for p in paths:
        chunks.append(Path(p).read_text(encoding="utf-8"))
    return "\n".join(chunks)

def main():
    ap = argparse.ArgumentParser(description="Export merchant table for manual labeling.")
    ap.add_argument("--txt-glob", default="synthetic_data/*.txt", help="Glob for training TXT files.")
    ap.add_argument("--out-csv", default="data/merchants_for_labeling.csv", help="Where to write the CSV.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.txt_glob))
    if not paths:
        raise SystemExit(f"No .txt found for pattern: {args.txt_glob}")

    raw = load_texts(paths)
    tx = parse_text_to_transactions(raw)
    if tx.empty:
        raise SystemExit("Parser produced 0 transactions. Check your input format.")

    tx2 = resolve_merchants(tx)

    # Build merchant-level features (for context columns)
    feat = build_feature_table(tx2)  # includes merchant_norm + cadence/amount stats

    # Add useful context columns for labeling
    agg = (
        tx2.groupby("merchant_norm")
           .agg(
               sample_descriptions=("description", lambda s: "; ".join(list(map(str, s))[:3])),
               brand=("brand", "first"),
               category=("category", "first"),
               count=("merchant_norm", "size"),
           )
           .reset_index()
    )

    df = feat.merge(agg, on="merchant_norm", how="left")

    # Move context columns to front, append a blank human_label column
    front = [
        "merchant_norm", "sample_descriptions", "brand", "category",
        "count", "span_days", "mean_amt", "cv",
        "is_weekly", "is_biweekly", "is_monthly", "is_quarterly", "is_yearly"
    ]
    for c in front:
        if c not in df.columns:
            df[c] = None
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]
    df["human_label"] = ""  # <-- YOU will fill 1 or 0

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] Wrote labeling CSV -> {out_path}")
    print("Open it, fill 'human_label' with 1 (subscription) or 0 (not), save, and then train with --labels-csv.")
    
if __name__ == "__main__":
    main()
