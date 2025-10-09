# frontend/app.py
import streamlit as st
import sys, os
from pathlib import Path
import pandas as pd

# project root import (so ml.* resolves)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.ocr_pipeline import extract_text_from_file
from ml.parse_transactions import parse_text_to_transactions
from ml.merchant_resolver import resolve_merchants
from ml.recurring import detect_recurring_subscriptions
from ml.anomalies import flag_amount_anomalies, flag_missed_cycles
from ml.score_subs import SubscriptionScorer
from ml.train_subs_models import train_from_transactions

st.set_page_config(page_title="SubTrackr", layout="wide")
st.sidebar.title("SubTrackr")
st.sidebar.markdown("Dashboard")
st.sidebar.markdown("Recurring Payments")
st.sidebar.markdown("Expenditure Breakdown")
st.sidebar.markdown("Your Uploads")
st.sidebar.markdown("Reminders")
st.sidebar.markdown("---")
st.sidebar.markdown("About • FAQ")

st.title("Welcome back!")
st.caption("Here’s your subscription overview for this month.")

uploaded_file = st.file_uploader("Upload Bank Statements (PDF, JPG, PNG)", type=["pdf","jpg","jpeg","png"])

# Choose which model to prefer when available
PREFERRED_MODEL = st.selectbox("Detection engine", ["Auto (prefer XGBoost)", "RandomForest", "XGBoost", "Heuristic only"], index=0)

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    ext = Path(uploaded_file.name).suffix.lower()
    temp_path = Path("data") / f"temp{ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"Processing: {uploaded_file.name}")
    raw_text = extract_text_from_file(str(temp_path)) or ""

    print(f"Raw text ----->{raw_text}")

    tx = parse_text_to_transactions(raw_text)
    if tx.empty:
        st.error("No transactions parsed from this file.")
        st.stop()

    # Resolve merchants (NLP)
    tx = resolve_merchants(tx)

    # ---------- Primary detection: ML if available (as requested) ----------
    rf_path, xgb_path = "models/rf_subscription.pkl", "models/xgb_subscription.pkl"
    used_mode = "heuristic"
    subs_ml = None

    if PREFERRED_MODEL != "Heuristic only" and (os.path.exists("models/rf_subscription.pkl") or os.path.exists("models/xgb_subscription.pkl")):
        try:
            # Model paths
            rf_path = "models/rf_subscription.pkl"
            xgb_path = "models/xgb_subscription.pkl"
            meta_path = "models/subs_meta.json"

            # Initialize both scorers with meta alignment + higher threshold for precision
            rf_scorer = SubscriptionScorer(rf_path, meta_path=meta_path, threshold=1)
            xgb_scorer = SubscriptionScorer(xgb_path, meta_path=meta_path, threshold=0.65)

            # Model selection logic
            if PREFERRED_MODEL in {"Auto (prefer XGBoost)", "XGBoost"} and os.path.exists(xgb_path):
                scorer, used_mode = xgb_scorer, "xgboost"
            elif PREFERRED_MODEL == "RandomForest" and os.path.exists(rf_path):
                scorer, used_mode = rf_scorer, "random_forest"
            elif os.path.exists(xgb_path):
                scorer, used_mode = xgb_scorer, "xgboost"
            else:
                scorer, used_mode = rf_scorer, "random_forest"

            subs_ml = scorer.score(tx)[["merchant_norm","brand","category","prob","is_subscription","count","mean_amt"]]

        except Exception as e:
            st.warning(f"Model scoring unavailable ({e}). Falling back to heuristics.")
            subs_ml = None

    # ---------- Heuristic fallback (always available) ----------
    subs_heur = detect_recurring_subscriptions(tx)
    subs_heur = flag_missed_cycles(subs_heur)

    # ---------- Decide what to show as the "main" subscriptions ----------
    if subs_ml is not None:
        main_subs = subs_ml.rename(columns={"prob":"score"})
    else:
        main_subs = subs_heur.assign(score=lambda d: d["is_subscription"].astype(int))

    # ---- Dashboard KPIs (simple placeholders; you’ll refine later) ----
    total_subs = int((main_subs["is_subscription"]==1).sum()) if not main_subs.empty else 0
    avg_amount = float(main_subs.loc[main_subs["is_subscription"]==1, "mean_amt"].mean() or 0.0)
    monthly_cost = avg_amount * total_subs
    yearly_cost = monthly_cost * 12
    potential_savings = monthly_cost * 0.4

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Subscriptions", total_subs)
    c2.metric("Monthly Cost", f"${monthly_cost:,.2f}")
    c3.metric("Yearly Cost", f"${yearly_cost:,.2f}")
    c4.metric("Potential Savings", f"${potential_savings:,.2f}")

    st.subheader("Recent Subscriptions")
    if not main_subs.empty and (main_subs["is_subscription"]==1).any():
        recent = (main_subs[main_subs["is_subscription"]==1]
                  .sort_values("score", ascending=False)
                  .head(5))[["merchant_norm","brand","category","mean_amt"]]
        recent = recent.rename(columns={"merchant_norm":"Merchant","mean_amt":"Amount","brand":"Brand","category":"Category"})
        st.dataframe(recent, use_container_width=True)
    else:
        st.info("No subscription candidates yet. More months of data improves detection.")

    # Raw text (collapsed)
    with st.expander("Raw extracted text", expanded=False):
        st.text_area("Text", raw_text, height=200)

    st.subheader("Parsed Transactions")
    st.dataframe(tx[["date","description","amount","merchant_norm","brand","category"]], use_container_width=True, height=320)

    st.subheader(f"Subscription Candidates ({'ML: ' + used_mode if subs_ml is not None else 'Heuristic baseline'})")
    if subs_ml is not None:
        view = main_subs[["merchant_norm","brand","category","count","mean_amt","score","is_subscription"]]
    else:
        view = subs_heur[["merchant_norm","brand","category","count","mean_amt","cv","cadence","last_date","next_expected","missed_cycle","is_subscription"]]
    st.dataframe(view, use_container_width=True)

    # Anomalies (amount spikes)
    tx_flagged = flag_amount_anomalies(tx)
    st.subheader("Transaction Anomalies (amount spikes)")
    st.dataframe(tx_flagged[["date","description","amount","merchant_norm","brand","category","amount_anomaly"]], use_container_width=True)

    # --------- Dev-only: train silently, no metrics shown ----------
    with st.expander("Developer: Train models on this data (silent)", expanded=False):
        if st.button("Train RF & XGB (no metrics in UI)"):
            try:
                train_from_transactions(tx, out_dir="models")
                st.success("Models trained and saved to ./models (metrics suppressed in UI).")
            except Exception as e:
                st.error(f"Training failed: {e}")
