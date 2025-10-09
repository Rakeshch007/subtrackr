# ml/parse_transactions.py
from __future__ import annotations
import re
from typing import List, Dict
from dateutil.parser import parse as dtparse
import pandas as pd

DATE_PAT = re.compile(
    r"\b("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"                                # 01/12/2025 or 1-2-25
    r"|"
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"                                 # 2025-01-12 or 2025.1.12
    r"|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"     # Jan, January, Aug, August
    r"[ -]\d{1,2},?[ -]?\d{2,4}"                                     # Jan 12, 2025  OR Jan-12 2025
    r"|"
    r"\d{1,2}[ -](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ -]\d{2,4}"  # 02 Aug 2025  OR 02-Aug-2025
    r")\b",
    re.IGNORECASE
)

AMOUNT_PAT = re.compile(
    r"(?P<sign>[-+])?\s*(?:USD|US\$|\$)?\s*(?P<val>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})|\d+(?:\.\d{1,2})?)\b"
)

def _to_float(num_str: str, sign: str | None) -> float:
    v = float(num_str.replace(",", ""))
    return -v if sign == "-" else v

def parse_text_to_transactions(text: str) -> pd.DataFrame:
    """
    Heuristic line parser: needs a date and a money-looking amount on the same line.
    Skips common header/footer lines.
    """
    HEADER_KEYWORDS = {
        "account holder", "account number", "statement period", "statement date",
        "date description", "amount ($)", "opening balance", "closing balance",
        "page", "subtotal", "total", "summary"
    }

    def is_header_line(ln: str) -> bool:
        low = ln.lower()
        return any(k in low for k in HEADER_KEYWORDS)

    def is_money_token(token: str) -> bool:
        # Accept if it has a currency symbol/label OR a decimal point.
        t = token.strip().lower()
        return ("$" in t or "usd" in t or "." in t)

    records: List[Dict] = []
    for ln in (l.strip() for l in (text or "").splitlines()):
        if not ln:
            continue
        if is_header_line(ln):
            continue

        dm = DATE_PAT.search(ln)
        if not dm:
            continue

        # Keep only "money-looking" amounts
        raw_amount_hits = list(AMOUNT_PAT.finditer(ln))
        ams = [m for m in raw_amount_hits if is_money_token(m.group(0))]
        if not ams:
            continue

        last = ams[-1]
        amt = _to_float(last.group("val"), last.group("sign"))

        # Extra guard: ignore absurd magnitudes from noise (tweak as needed)
        if abs(amt) > 1_000_000:
            continue

        try:
            d = dtparse(dm.group(0), dayfirst=False, yearfirst=False).date()
        except Exception:
            continue

        s, e = last.span()
        desc = (ln[:dm.start()] + ln[dm.end():s] + ln[e:]).strip(" -:|•\t")
        if not desc:
            desc = "Transaction"

        records.append({
            "date": d,
            "description": desc,
            "amount": amt,
            "currency": "USD",
            "type": "debit" if amt < 0 else "credit"
        })

    df = pd.DataFrame.from_records(records, columns=["date","description","amount","currency","type"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.sort_values(["date","description"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df



if __name__ == "__main__":
    print("🔍 Testing parse_text_to_transactions...")

    sample_text = """
  Sample Bank Statement
Account Holder: John Doe
Account Number: XXXX-1234
Statement Period: 01-Aug-2025 to 31-Aug-2025
Date Description Amount ($)
02-Aug-2025 NETFLIX.COM Subscription 15.99
05-Aug-2025 Spotify Premium 9.99
10-Aug-2025 Amazon Marketplace 120.49
15-Aug-2025 LinkedIn Premium 29.99
20-Aug-2025 Adobe Creative Cloud 52.99
25-Aug-2025 Grocery Store 200.00
    """

    df = parse_text_to_transactions(sample_text)
    if df.empty:
        print("No transactions parsed.")
    else:
        print(df.to_string(index=False))
