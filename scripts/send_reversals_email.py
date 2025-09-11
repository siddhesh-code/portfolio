#!/usr/bin/env python3
"""
Minimal 'Reversals summary' email sender (on-demand).

Usage:
  python scripts/send_reversals_email_min.py --to you@gmail.com --universe n50 --top 12

SMTP config via ENV (typical for Gmail):
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=you@gmail.com
  SMTP_PASS=app_password
  MAIL_FROM=you@gmail.com
"""

from __future__ import annotations

import os
import sys
import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Make project imports work when run from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import NIFTY50_SYMBOLS, NIFTY500_SYMBOLS, POSITIONS # reuse your existing lists
from data_fetcher import get_nifty50_data
from stock_analyzer import format_stock_analysis
from reversals import rank_reversals


def resolve_universe(key: str) -> list[str]:
    u = (key or 'n50').lower()
    if u == 'n500':
        return list(NIFTY500_SYMBOLS)
    elif u == 'watch':
        return list(POSITIONS)
    return list(NIFTY50_SYMBOLS)


def build_html_simple(rev: dict) -> str:
    """Return a compact HTML string (no fancy styling)."""
    def rows(items):
        out = []
        for it in items:
            sym = it.get('symbol', '-')
            name = it.get('name') or sym
            price = it.get('price', 0.0)
            score = float(it.get('score', 0.0))
            out.append(
                f"<tr><td>{name}<br><span style='color:#666'>{sym}</span></td>"
                f"<td align='right'>₹{price:,.2f}</td>"
                f"<td align='right'>{score:.0f}</td></tr>"
            )
        return "".join(out) if out else "<tr><td colspan='3' style='color:#666'>None</td></tr>"

    bull = rows(rev.get('bullish', []))
    bear = rows(rev.get('bearish', []))
    html = (
        "<!doctype html><html><body>"
        "<h3>Reversals — Top Bullish</h3>"
        "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px'>"
        "<thead><tr><th align='left'>Symbol</th><th align='right'>Price</th><th align='right'>Score</th></tr></thead>"
        f"<tbody>{bull}</tbody></table>"
        "<br>"
        "<h3>Reversals — Top Bearish</h3>"
        "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px'>"
        "<thead><tr><th align='left'>Symbol</th><th align='right'>Price</th><th align='right'>Score</th></tr></thead>"
        f"<tbody>{bear}</tbody></table>"
        "<p style='color:#666;font-size:12px'>Automated summary. Not investment advice.</p>"
        "</body></html>"
    )
    return html


def build_text_simple(rev: dict) -> str:
    """Plain-text fallback for clients that prefer it."""
    def lines(title, items):
        out = [title]
        if not items:
            out.append("  None")
            return out
        for it in items:
            sym = it.get('symbol', '-')
            name = it.get('name') or sym
            price = it.get('price', 0.0)
            score = float(it.get('score', 0.0))
            out.append(f"  {sym} ({name}): ₹{price:,.2f}, Score {score:.0f}")
        return out

    bull = lines("Top Bullish", rev.get('bullish', []))
    bear = lines("Top Bearish", rev.get('bearish', []))
    return "\n".join(bull + [""] + bear + ["", "Automated summary. Not investment advice."])


def send_mail(subject: str, html_body: str, text_body: str, to_addrs: list[str]) -> None:
    """Minimal SMTP sender. No prints unless error."""
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    pwd  = os.getenv("SMTP_PASS", "")
    mail_from = os.getenv("MAIL_FROM", user)

    if not (mail_from and to_addrs and user and pwd):
        raise RuntimeError("Missing SMTP creds or recipients. Set SMTP_HOST/PORT/USER/PASS and MAIL_FROM, and pass --to.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = ", ".join(to_addrs)
    if text_body:
        msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(host, port, timeout=20) as s:
        s.ehlo()
        if port in (587, 25):
            s.starttls()
            s.ehlo()
        s.login(user, pwd)
        s.sendmail(mail_from, to_addrs, msg.as_string())


def main():
    ap = argparse.ArgumentParser(description="Send a simple reversals email (on-demand).")
    ap.add_argument("--to", default="siddheshdhoot99@gmail.com", help="Recipient email (comma-separated for multiple)")
    ap.add_argument("--universe", default="n500", choices=["n50", "n500" ,"watch"], help="Symbol universe")
    ap.add_argument("--top", type=int, default=10, help="Top N per side")
    
    args = ap.parse_args()

    symbols = resolve_universe(args.universe)
    raw = get_nifty50_data(symbols)
    formatted = format_stock_analysis(raw)
    rev = rank_reversals(raw, formatted, top_n=args.top)

    html = build_html_simple(rev)
    text = build_text_simple(rev)
    subject = f"Reversals {args.universe.upper()} (Top {args.top})"
    to_list = [x.strip() for x in args.to.split(",") if x.strip()]

    send_mail(subject, html, text, to_list)


if __name__ == "__main__":
    main()
