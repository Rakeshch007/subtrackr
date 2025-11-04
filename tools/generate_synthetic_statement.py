#!/usr/bin/env python
"""
Generate a synthetic 6-month bank statement with diverse recurring subscriptions
(weekly/monthly/quarterly/yearly with jitter, missed cycles, plan changes) and
noisy daily spend.

SIGN CONVENTION:
  - Debits (charges)  -> NEGATIVE amounts
  - Credits (refunds) -> POSITIVE amounts

Outputs:
  - synthetic_statement_<seed>_6mo.txt  (for your parser)
  - synthetic_statement_<seed>_6mo.csv  (structured table)
  - synthetic_statement_<seed>_6mo.pdf  (for upload & OCR testing)

Run:
  python tools/generate_synthetic_statement.py
"""

from __future__ import annotations
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ------------- CONFIG -------------
MONTHS_BACK = 12
OUT_DIR = Path("synthetic_data")
ACCOUNT_HOLDER = "John Doe"
# New random seed each run for uniqueness; set e.g. SEED=1234 to reproduce.
SEED = random.randint(1000, 999999)
# ----------------------------------


# ---------- Helpers ----------
def fmt_date(d: date) -> str:
    """Match your parser format: DD-Mon-YYYY, e.g., 02-Aug-2025."""
    return d.strftime("%d-%b-%Y")

def first_day_n_months_ago(n: int) -> date:
    today = date.today()
    return (today - timedelta(days=30*n)).replace(day=1)

# ---------- Subscription pools ----------
# (name, typical_amount, cadence_label)
# cadence_label in {"weekly","monthly","quarterly","yearly"} (we'll map to days)
SUB_BRANDS_POOL: List[Tuple[str, float, str]] = [
        # Entertainment / Streaming
    ("Netflix", 17.99, "monthly"), ("Disney+", 12.00, "monthly"),
    ("Max", 10.99, "monthly"), ("Hulu", 7.99, "monthly"),
    ("Prime Video", 8.99, "monthly"), ("Apple TV+", 12.99, "monthly"),
    ("Paramount+", 7.99, "monthly"), ("Peacock", 10.99, "monthly"),
    ("Crunchyroll", 7.99, "monthly"), ("ESPN+", 10.99, "monthly"),
    ("DAZN", 19.99, "monthly"),
    # Music / Audio
    ("Spotify Premium", 10.99, "monthly"), ("Apple Music", 10.99, "monthly"),
    ("YouTube Premium", 13.99, "monthly"), ("YouTube Music", 10.99, "monthly"),
    ("TIDAL", 10.99, "monthly"), ("Audible", 14.95, "monthly"),
    ("SiriusXM", 11.99, "monthly"),
    # News / Magazines
    ("NYTimes", 25.00, "monthly"), ("WSJ", 38.99, "monthly"),
    ("Washington Post", 12.00, "monthly"), ("The Economist", 19.00, "monthly"),
    ("Bloomberg", 34.99, "monthly"), ("Financial Times", 39.00, "monthly"),
    ("Medium", 5.00, "monthly"),
    # Gaming
    ("PlayStation Plus", 9.99, "monthly"), ("Xbox Game Pass", 10.99, "monthly"),
    ("Nintendo Switch Online", 3.99, "monthly"), ("Twitch Turbo", 11.99, "monthly"),
    # Productivity / Software
    ("Adobe Creative Cloud", 59.99, "monthly"), ("Photoshop", 22.99, "monthly"),
    ("Microsoft 365", 6.99, "monthly"), ("Notion", 10.00, "monthly"),
    ("Slack", 8.75, "monthly"), ("Zoom Pro", 14.99, "monthly"),
    ("Canva Pro", 14.99, "monthly"), ("Figma", 15.00, "monthly"),
    ("Asana", 13.49, "monthly"), ("Monday.com", 12.00, "monthly"),
    ("Evernote", 10.83, "monthly"), ("Grammarly", 30.00, "monthly"),
    ("Dropbox", 11.99, "monthly"), ("Box", 15.00, "monthly"),
    ("OneDrive", 6.99, "monthly"),
    # Developer / Cloud
    ("GitHub", 4.00, "monthly"), ("GitLab", 29.00, "monthly"),
    ("Bitbucket", 15.00, "monthly"), ("DigitalOcean", 4.00, "monthly"),
    ("Linode", 5.00, "monthly"), ("Heroku", 5.00, "monthly"),
    ("Vercel", 20.00, "monthly"), ("Netlify", 19.00, "monthly"),
    ("Render", 7.00, "monthly"), ("Cloudflare", 5.00, "monthly"),
    ("AWS", 25.00, "monthly"), ("GCP", 25.00, "monthly"),
    ("Azure", 25.00, "monthly"),
    # Storage / Backup
    ("Google One", 1.99, "monthly"), ("Apple iCloud", 0.99, "monthly"),
    ("Backblaze", 7.00, "monthly"), ("IDrive", 2.95, "monthly"),
    ("MEGA", 6.07, "monthly"),
    # Security / VPN / Passwords
    ("1Password", 2.99, "monthly"), ("LastPass", 3.00, "monthly"),
    ("Dashlane", 4.99, "monthly"), ("Malwarebytes", 4.99, "monthly"),
    ("NordVPN", 12.99, "monthly"), ("ExpressVPN", 12.95, "monthly"),
    ("Surfshark", 2.49, "monthly"), ("Proton VPN", 4.99, "monthly"),
    # Education / Learning
    ("Coursera Plus", 59.00, "monthly"), ("Udemy Membership", 29.99, "monthly"),
    ("Skillshare", 15.00, "monthly"), ("LinkedIn Learning", 39.99, "monthly"),
    ("Duolingo Plus", 6.99, "monthly"), ("Babbel", 12.95, "monthly"),
    ("Brilliant", 12.49, "monthly"), ("Chegg", 19.95, "monthly"),
    # Fitness / Wellness
    ("Peloton", 24.00, "monthly"), ("Fitbit Premium", 9.99, "monthly"),
    ("Strava", 11.99, "monthly"), ("MyFitnessPal", 19.99, "monthly"),
    ("Headspace", 12.99, "monthly"), ("Calm", 14.99, "monthly"),
    # Finance / Budgeting
    ("YNAB", 14.99, "monthly"), ("QuickBooks", 30.00, "monthly"),
    ("Xero", 13.00, "monthly"), ("Rocket Money", 4.00, "monthly"),
    # Shopping / Memberships
    ("Amazon Prime", 14.99, "monthly"), ("Walmart+", 12.95, "monthly"),
    ("Costco Membership", 60.00, "yearly"), ("Sam's Club", 50.00, "yearly"),
    # Mobility
    ("Uber One", 9.99, "monthly"), ("Lyft Pink", 9.99, "monthly"),
    # Communication / Email
    ("Google Workspace", 6.00, "monthly"), ("Proton Mail", 4.99, "monthly"),
    ("Fastmail", 3.00, "monthly"),
    # AI / Creative
    ("ChatGPT Plus", 20.00, "monthly"), ("Claude Pro", 20.00, "monthly"),
    ("Midjourney", 10.00, "monthly"), ("GitHub Copilot", 10.00, "monthly"),
    ("Jasper AI", 49.00, "monthly"),
    # Quarterly/Yearly extras
    ("Antivirus Suite", 39.99, "yearly"), ("Cloud Backup Pro", 24.00, "quarterly")
]

NOISE_MERCHANTS = [
    "Amazon Marketplace","Grocery Store","Walmart","Target","Starbucks","McDonald's","Domino's Pizza",
    "Subway","Shell Gas Station","Chevron Gas","Uber Trip","Lyft Ride","Apple Store","Best Buy",
    "Pharmacy Express","Local Diner","Hardware Depot","Home Improvement","IKEA","Electronics Hub",
    "Book Store","Cinema City","Gas Station","KFC","Burger King","PayPal Transfer","Venmo Payment",
    "ATM Withdrawal","Mobile Recharge","Electric Bill","Water Bill","Internet Payment",
    "Credit Card Payment","Insurance Premium","Medical Center","Dental Clinic","Car Wash",
    "Gym Day Pass","Hotel Booking","Flight Ticket","Train Ticket","Amazon Refund","Spotify Refund",
    "Movie Tickets","Lottery","Charity Donation","Tax Refund","Government Fees","Parking Fine",
    "Pet Store","Toy Shop","Thrift Store","Farmers Market","Convenience Store","Liquor Store",
    "Bakery","Butcher Shop","Fish Market","Garden Center","Shoe Outlet"
]

CADENCE_TO_DAYS = {"weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365}

# ---------- Generators ----------
def gen_subscription_rows(start_date: date, months: int, rng: random.Random) -> List[dict]:
    """Generate recurring subscriptions with missed cycles and plan changes.
       Debits -> NEGATIVE amounts."""
    rows = []
    # Pick 10–20 subscriptions for this synthetic user
    chosen = rng.sample(SUB_BRANDS_POOL, k=rng.randint(10, 20))

    for name, base_amt, cadence in chosen:
        anchor = start_date.replace(day=rng.randint(3, 25))
        cur = anchor

        # missed cycles: 0–1 missed months for realism
        missed_indices = set()
        if cadence in {"monthly", "weekly"} and rng.random() < 0.25:
            missed_indices.add(rng.randint(0, max(0, months-1)))

        # plan change month (bump/drop)
        plan_change_idx = rng.randint(0, max(0, months-1)) if rng.random() < 0.35 else None
        plan_delta = rng.uniform(-2.0, 5.0)

        for i in range(months):
            # advance by cadence
            cur = cur + timedelta(days=CADENCE_TO_DAYS.get(cadence, 30))
            # date jitter
            jitter_low, jitter_high = (-1, 1) if cadence == "weekly" else (-3, 3)
            charge_date = cur + timedelta(days=rng.randint(jitter_low, jitter_high))

            if i in missed_indices:
                continue

            # base jitter
            amount = base_amt + rng.uniform(-0.7, 1.2)
            # occasional tax/fee surcharges
            if rng.random() < 0.10:
                amount += rng.uniform(0.5, 2.0)
            # plan change
            if plan_change_idx is not None and i >= plan_change_idx:
                amount += plan_delta

            # DEBIT -> NEGATIVE
            amount = -round(max(0.99, amount), 2)

            rows.append({
                "date": charge_date,
                "description": f"{name} Subscription",
                "amount": amount,            # negative
                "currency": "USD",
                "type": "debit"
            })

    return rows

def gen_noise_rows(start_date: date, months: int, rng: random.Random) -> List[dict]:
    """Generate daily noise transactions (debits negative, credits positive)."""
    rows = []
    end_date = start_date + timedelta(days=30*months + 5)
    d = start_date
    while d <= end_date:
        # 0–3 random purchases a day
        for _ in range(rng.randint(0, 3)):
            desc = rng.choice(NOISE_MERCHANTS)
            # decide if refund/credit (positive) or purchase (negative)
            is_refund = rng.random() < 0.08
            magnitude = rng.uniform(5.0, 350.0) if not is_refund else rng.uniform(5.0, 150.0)
            amount = round(magnitude, 2)
            if is_refund:
                desc += " REFUND"
                sign_amount = amount       # CREDIT -> POSITIVE
                tx_type = "credit"
            else:
                sign_amount = -amount      # DEBIT -> NEGATIVE
                tx_type = "debit"

            # occasional duplicate-like charges same day (same sign)
            if rng.random() < 0.02:
                rows.append({
                    "date": d,
                    "description": desc,
                    "amount": sign_amount,
                    "currency": "USD",
                    "type": tx_type
                })

            rows.append({
                "date": d,
                "description": desc,
                "amount": sign_amount,
                "currency": "USD",
                "type": tx_type
            })
        d += timedelta(days=1)
    return rows

# ---------- Statement build & PDF ----------
def build_statement_text(df: pd.DataFrame, acct_last4: int, first_day: date, last_day: date) -> str:
    header = [
        "Sample Bank Statement",
        f"Account Holder: {ACCOUNT_HOLDER}",
        f"Account Number: XXXX-{acct_last4}",
        f"Statement Period: {fmt_date(first_day)} to {fmt_date(last_day)}",
        "Date Description Amount ($)"
    ]
    lines = header + [f"{fmt_date(r['date'])} {r['description']} {r['amount']:.2f}"
                      for r in df.to_dict(orient='records')]
    return "\n".join(lines)

def render_pdf(text_lines: List[str], pdf_path: Path):
    lines_per_page = 58
    pages = [text_lines[i:i+lines_per_page] for i in range(0, len(text_lines), lines_per_page)]
    with PdfPages(pdf_path) as pdf:
        for i, page_lines in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            y = 0.97; line_height = 0.016
            for ln in page_lines:
                ax.text(0.05, y, ln, family="DejaVu Sans Mono", fontsize=9, va="top")
                y -= line_height
            ax.text(0.5, 0.02, f"Page {i}", ha="center", va="bottom",
                    fontsize=8, family="DejaVu Sans Mono")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

# ---------- Main ----------
def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    first_day = first_day_n_months_ago(MONTHS_BACK)
    last_day = date.today()

    subs = gen_subscription_rows(first_day, MONTHS_BACK, rng)
    noise = gen_noise_rows(first_day, MONTHS_BACK, rng)

    df = pd.DataFrame(subs + noise).sort_values("date").reset_index(drop=True)
    # Keep within window
    df = df[(df["date"] >= first_day) & (df["date"] <= last_day)].reset_index(drop=True)

    acct_last4 = rng.randint(1000, 9999)
    statement_text = build_statement_text(df, acct_last4, first_day, last_day)

    txt_path = OUT_DIR / f"synthetic_statement_{SEED}_6mo.txt"
    csv_path = OUT_DIR / f"synthetic_statement_{SEED}_6mo.csv"
    pdf_path = OUT_DIR / f"synthetic_statement_{SEED}_6mo.pdf"

    txt_path.write_text(statement_text, encoding="utf-8")
    df_out = df.copy(); df_out["date"] = df_out["date"].astype(str)
    df_out.to_csv(csv_path, index=False)
    render_pdf(statement_text.splitlines(), pdf_path)

    print(f"[✅] Seed: {SEED}")
    print(f"[ok] TXT -> {txt_path}")
    print(f"[ok] CSV -> {csv_path}")
    print(f"[ok] PDF -> {pdf_path}")
    print("Upload the PDF to SubTrackr (OCR → Parse → NLP → Detection), or feed the TXT to the parser directly.")

if __name__ == "__main__":
    main()
