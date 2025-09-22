"""SIP planner and alerting tool.

Loads configuration from sip_config.json, fetches market data, builds a monthly
buy plan, renders console tables, stores historical snapshots, and optionally
pushes notifications (email / Slack / Telegram).
"""
import argparse
import json
import math
import os
import smtplib
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta
from email.message import EmailMessage
from rich.console import Console
from rich.table import Table

console = Console()
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("sip_config.json")
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "sip_prices.json"
DEFAULT_NOTES_PATH = Path(__file__).resolve().with_name("notes.txt")

DEFAULT_CONFIG = {
    "budget": {
        "monthly_budget": 80_000,
        "core_pct": 0.60,
        "dip_pct": 0.25,
        "buffer_pct": 0.15,
    },
    "holdings": {
        "DCXINDIA.NS": {"qty": 30, "avg_price": 280.0},
        "MTARTECH.NS": {"qty": 5, "avg_price": 1777.88},
        "SBCL.NS": {"qty": 13, "avg_price": 552.45},
        "TDPOWERSYS.NS": {"qty": 15, "avg_price": 538.25},
        "WABAG.NS": {"qty": 6, "avg_price": 1549.80},
    },
    "triggers": {
        "price_drop_primary": 0.12,
        "price_drop_strong": 0.15,
        "nifty_buffer_partial": -0.05,
        "nifty_buffer_full": -0.10,
    },
    "data": {
        "cache_path": str(DEFAULT_CACHE_PATH),
        "history_days": 365,
        "retry_attempts": 3,
        "retry_backoff": 1.6,
    },
    "reporting": {
        "output_dir": "reports",
        "history_file": "reports/sip_history.json",
        "save_markdown": True,
        "save_csv": True,
    },
    "notifications": {
        "email": {"enabled": True},
        "slack": {"enabled": False, "webhook_url": ""},
        "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
    },
    "fundamentals_flags": "fundamentals_flags.json",
}


def deep_update(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def make_json_safe(value):
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.generic):  # numpy scalar
            val = value.item()
            if isinstance(val, float) and math.isnan(val):
                return None
            return val
    except Exception:
        pass
    try:
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Optional[str]) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        try:
            file_cfg = load_json(config_path)
            cfg = deep_update(cfg, file_cfg)
        except Exception as e:
            console.print(f"[red]Failed to read config {config_path}: {e}[/red]")
    else:
        console.print(f"[yellow]Config file {config_path} not found. Using defaults.[/yellow]")
    return cfg


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class MarketDataFetcher:
    def __init__(self, cache_path: Path, offline: bool = False,
                 retries: int = 3, backoff: float = 1.5) -> None:
        self.cache_path = cache_path
        self.offline = offline
        self.retries = max(1, retries)
        self.backoff = max(1.0, backoff)
        self.price_cache: Dict[str, dict] = {}
        self.history_cache: Dict[str, dict] = {}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            self.price_cache = {}
            self.history_cache = {}
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.price_cache = data.get("prices", {})
            self.history_cache = data.get("history", {})
        except Exception:
            self.price_cache = {}
            self.history_cache = {}

    def _save_cache(self) -> None:
        data = {
            "prices": self.price_cache,
            "history": self.history_cache,
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _is_stale(self, timestamp: str, days: int = 0) -> bool:
        if not timestamp:
            return True
        try:
            ts = datetime.fromisoformat(timestamp)
        except Exception:
            return True
        if days == 0:
            return ts.date() != date.today()
        return datetime.utcnow() - ts > timedelta(days=days)

    def get_latest_price(self, ticker: str) -> float:
        cached = self.price_cache.get(ticker)
        if cached and not self._is_stale(cached.get("timestamp", "")):
            return float(cached["price"])
        if self.offline:
            if cached:
                return float(cached["price"])
            raise RuntimeError(f"No cached price for {ticker} in offline mode")
        price = self._fetch_price(ticker)
        self.price_cache[ticker] = {
            "price": price,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._save_cache()
        return price

    def _fetch_price(self, ticker: str) -> float:
        delay = 0.0
        last_err = None
        for attempt in range(self.retries):
            try:
                t = yf.Ticker(ticker)
                fast = getattr(t, "fast_info", None)
                if fast:
                    val = getattr(fast, "last_price", None) or fast.get("last_price")
                    if val and val > 0:
                        return float(val)
                hist = t.history(period="5d", interval="1d", auto_adjust=False)
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception as e:
                last_err = e
            delay = self.backoff ** attempt
            if delay:
                time.sleep(delay)
        raise RuntimeError(f"Unable to fetch price for {ticker}: {last_err}")

    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        key = f"{period}_{interval}"
        ticker_hist = self.history_cache.get(ticker, {})
        cached = ticker_hist.get(key)
        if cached and not self._is_stale(cached.get("timestamp", "")):
            df = pd.DataFrame(cached["data"])
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_convert(None)
            return df
        if self.offline:
            if cached:
                df = pd.DataFrame(cached["data"]) 
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")
                    if getattr(df.index, "tz", None) is not None:
                        df.index = df.index.tz_convert(None)
                return df
            raise RuntimeError(f"No cached history for {ticker} in offline mode")
        df = self._fetch_history(ticker, period=period, interval=interval)
        if df.empty:
            raise RuntimeError(f"Empty history for {ticker}")
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        reset_df = df.reset_index()
        if "Date" in reset_df.columns:
            reset_df["Date"] = reset_df["Date"].astype(str)
        records = reset_df.to_dict(orient="records")
        ticker_hist[key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": records,
        }
        self.history_cache[ticker] = ticker_hist
        self._save_cache()
        return df

    def _fetch_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        delay = 0.0
        last_err = None
        for attempt in range(self.retries):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period=period, interval=interval, auto_adjust=False)
                if not hist.empty:
                    if getattr(hist.index, "tz", None) is not None:
                        hist.index = hist.index.tz_convert(None)
                    return hist
            except Exception as e:
                last_err = e
            delay = self.backoff ** attempt
            if delay:
                time.sleep(delay)
        raise RuntimeError(f"Unable to fetch history for {ticker}: {last_err}")


@dataclass
class TickerPlan:
    ticker: str
    quantity: float
    avg_price: float
    cost_basis: float
    latest_price: float
    prev_month_close: Optional[float]
    drop_pct: Optional[float]
    fund_flag: bool
    dip_trigger: bool
    strong_trigger: bool
    metrics: dict
    allocations: dict
    valuation: dict


def compute_prev_month_close(history: pd.DataFrame) -> Optional[float]:
    if history.empty:
        return None
    today = date.today()
    first_of_month = date(today.year, today.month, 1)
    prev_month_end = first_of_month - timedelta(days=1)
    subset = history.loc[:prev_month_end.strftime("%Y-%m-%d")]
    if subset.empty:
        return None
    return float(subset["Close"].iloc[-1])


def compute_month_to_date_change(history: pd.DataFrame) -> Optional[float]:
    if history.empty:
        return None
    today = date.today()
    month_start = date(today.year, today.month, 1)
    subset = history.loc[month_start:]
    if subset.empty:
        return None
    open_price = float(subset["Close"].iloc[0])
    last_price = float(subset["Close"].iloc[-1])
    if open_price == 0:
        return None
    return (last_price / open_price) - 1.0


def compute_metrics(history: pd.DataFrame) -> dict:
    metrics = {}
    if history.empty:
        return metrics
    closes = history["Close"].dropna()
    if closes.empty:
        return metrics
    metrics["sma20"] = float(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else None
    metrics["sma50"] = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
    metrics["vol20"] = float(closes.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252)) if len(closes) >= 21 else None
    metrics["high_52w"] = float(closes.tail(252).max()) if len(closes) >= 10 else None
    metrics["low_52w"] = float(closes.tail(252).min()) if len(closes) >= 10 else None
    if metrics.get("high_52w"):
        metrics["drawdown"] = (closes.iloc[-1] / metrics["high_52w"]) - 1.0
    else:
        metrics["drawdown"] = None
    return metrics


def load_fundamental_flags(path: Optional[str]) -> Dict[str, bool]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return {k: bool(v) for k, v in load_json(p).items()}
    except Exception:
        return {}


def allocate_budget(total_budget: int, weights: Dict[str, float]) -> Dict[str, int]:
    if total_budget <= 0 or not weights:
        return {t: 0 for t in weights}
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 0:
        even = total_budget // len(weights)
        alloc = {t: even for t in weights}
        remainder = total_budget - sum(alloc.values())
        tickers = list(weights.keys())
        for i in range(remainder):
            alloc[tickers[i % len(weights)]] += 1
        return alloc
    raw = {}
    floors = {}
    remainders = []
    for ticker, weight in weights.items():
        proportion = max(weight, 0.0) / total
        value = total_budget * proportion
        raw[ticker] = value
        floor_val = math.floor(value)
        floors[ticker] = floor_val
        remainders.append((value - floor_val, ticker))
    allocated = sum(floors.values())
    remainder = total_budget - allocated
    remainders.sort(reverse=True)
    for _, ticker in remainders:
        if remainder <= 0:
            break
        floors[ticker] += 1
        remainder -= 1
    return floors


def compute_plan(config: dict, fetcher: MarketDataFetcher, args) -> Tuple[List[TickerPlan], dict, dict, List[str]]:
    holdings = config.get("holdings", {})
    if args.only:
        tickers = {t.strip().upper(): True for t in args.only.split(",") if t.strip()}
        holdings = {t: data for t, data in holdings.items() if t.upper() in tickers}
    if not holdings:
        raise RuntimeError("No holdings configured")

    fund_flags = load_fundamental_flags(config.get("fundamentals_flags"))
    budget_cfg = config["budget"]
    monthly_budget = int(budget_cfg.get("monthly_budget", 0))
    if args.budget:
        monthly_budget = args.budget
    core_budget = int(round(monthly_budget * budget_cfg.get("core_pct", 0.0)))
    dip_budget = int(round(monthly_budget * budget_cfg.get("dip_pct", 0.0)))
    buffer_budget = int(round(monthly_budget * budget_cfg.get("buffer_pct", 0.0)))
    correction = monthly_budget - (core_budget + dip_budget + buffer_budget)
    buffer_budget += correction

    weights = {}
    for ticker, details in holdings.items():
        qty = float(details.get("qty", 0.0))
        avg_price = float(details.get("avg_price", 0.0))
        weight = qty * avg_price
        if weight <= 0:
            weight = max(qty, 1.0)
        weights[ticker] = weight

    core_alloc = allocate_budget(core_budget, weights)
    dip_alloc = allocate_budget(dip_budget, weights)

    data_cfg = config.get("data", {})
    history_days = int(data_cfg.get("history_days", 365))

    plans: List[TickerPlan] = []
    actions: List[str] = []
    totals = {
        "core_budget": core_budget,
        "dip_budget": dip_budget,
        "buffer_budget": buffer_budget,
        "monthly_budget": monthly_budget,
        "deploy_now": 0,
        "carry_forward": 0,
        "market_value": 0.0,
        "cost_basis": 0.0,
        "unrealized_pnl": 0.0,
    }

    for ticker, details in holdings.items():
        qty = float(details.get("qty", 0.0))
        avg_price = float(details.get("avg_price", 0.0))
        cost = qty * avg_price
        latest_price = fetcher.get_latest_price(ticker)
        history = fetcher.get_history(ticker, period=f"{history_days}d")
        history = history.sort_index()
        prev_close = compute_prev_month_close(history)
        drop_pct = None
        price_trigger = False
        strong_trigger = False
        trig_cfg = config.get("triggers", {})
        if prev_close:
            drop_pct = (latest_price / prev_close) - 1.0
            if drop_pct <= -float(trig_cfg.get("price_drop_primary", 0.12)):
                price_trigger = True
            if drop_pct <= -float(trig_cfg.get("price_drop_strong", 0.15)):
                strong_trigger = True
        fund_flag = bool(fund_flags.get(ticker, False))
        metrics = compute_metrics(history)
        mtd_change = compute_month_to_date_change(history)
        metrics["mtd_change"] = mtd_change

        valuation = {
            "market_value": qty * latest_price,
            "pnl": qty * latest_price - cost,
            "pnl_pct": ((latest_price / avg_price) - 1.0) if avg_price else None,
        }
        allocations = {
            "core": core_alloc.get(ticker, 0),
            "dip": dip_alloc.get(ticker, 0),
        }
        deploy_now = 0
        if allocations["dip"] > 0 and (price_trigger or fund_flag):
            deploy_now += allocations["dip"]
            action_label = "strong dip" if strong_trigger and price_trigger else "trigger"
            actions.append(f"Dip pot: {ticker} → deploy ₹{allocations['dip']:,} ({action_label})")
        else:
            totals["carry_forward"] += allocations["dip"]
        totals["deploy_now"] += deploy_now
        totals["market_value"] += valuation["market_value"]
        totals["cost_basis"] += cost
        totals["unrealized_pnl"] += valuation["pnl"]

        plan = TickerPlan(
            ticker=ticker,
            quantity=qty,
            avg_price=avg_price,
            cost_basis=cost,
            latest_price=latest_price,
            prev_month_close=prev_close,
            drop_pct=drop_pct,
            fund_flag=fund_flag,
            dip_trigger=price_trigger or fund_flag,
            strong_trigger=strong_trigger,
            metrics=metrics,
            allocations=allocations,
            valuation=valuation,
        )
        plans.append(plan)

    nifty_change = None
    try:
        nifty_history = fetcher.get_history("^NSEI", period=f"{history_days}d")
        nifty_history = nifty_history.sort_index()
        nifty_change = compute_month_to_date_change(nifty_history)
    except Exception:
        pass

    trig_cfg = config.get("triggers", {})
    buffer_action = "HOLD"
    buffer_to_deploy = 0
    if nifty_change is not None:
        if nifty_change <= trig_cfg.get("nifty_buffer_full", -0.10):
            buffer_action = "DEPLOY FULL (broad -10% MTD)"
            buffer_to_deploy = buffer_budget
        elif nifty_change <= trig_cfg.get("nifty_buffer_partial", -0.05):
            buffer_action = "DEPLOY 50% (broad -5% MTD)"
            buffer_to_deploy = buffer_budget * 0.5
    if buffer_to_deploy > 0:
        actions.append(f"Buffer: {buffer_action} → deploy ₹{buffer_to_deploy:,.0f}")

    buffer_summary = {
        "buffer_budget": buffer_budget,
        "buffer_action": buffer_action,
        "buffer_to_deploy": buffer_to_deploy,
        "nifty_mtd": nifty_change,
    }
    totals["overall_return_pct"] = (totals["market_value"] / totals["cost_basis"] - 1.0) if totals["cost_basis"] else None
    return plans, totals, buffer_summary, actions


def render_console(plans: List[TickerPlan], totals: dict, buffer_summary: dict, notes: Optional[str]) -> None:
    today = date.today().isoformat()
    console.print(f"[bold cyan]SIP Planner — {today}[/bold cyan]")

    d1, d2 = next_buy_dates()
    table = Table(title="Core SIP — 2 tranches (4th & ~15th)")
    table.add_column("Ticker", style="cyan")
    table.add_column("Tranche Date", justify="center")
    table.add_column("Amount ₹", justify="right")
    for plan in plans:
        amt = plan.allocations["core"]
        if amt <= 0:
            continue
        table.add_row(plan.ticker, d1.isoformat(), f"{amt/2:,.0f}")
        table.add_row(plan.ticker, d2.isoformat(), f"{amt/2:,.0f}")
    console.print(table)

    dip_tbl = Table(title="Dip Pot — triggers & diagnostics")
    dip_tbl.add_column("Ticker", style="cyan")
    dip_tbl.add_column("Dip ₹", justify="right")
    dip_tbl.add_column("Last ₹", justify="right")
    dip_tbl.add_column("Prev Mo ₹", justify="right")
    dip_tbl.add_column("Δ vs Prev Mo", justify="right")
    dip_tbl.add_column("SMA20", justify="right")
    dip_tbl.add_column("SMA50", justify="right")
    dip_tbl.add_column("Drawdown", justify="right")
    dip_tbl.add_column("Action", style="green")

    for plan in plans:
        drop_pct = plan.drop_pct * 100 if plan.drop_pct is not None else None
        metrics = plan.metrics
        action = "HOLD"
        if plan.allocations["dip"] > 0 and plan.dip_trigger:
            action = "BUY (strong dip)" if plan.strong_trigger else "BUY"
        dip_tbl.add_row(
            plan.ticker,
            f"{plan.allocations['dip']:,.0f}",
            f"{plan.latest_price:,.2f}",
            f"{plan.prev_month_close:,.2f}" if plan.prev_month_close else "n/a",
            f"{drop_pct:.1f}%" if drop_pct is not None else "n/a",
            f"{metrics.get('sma20', float('nan')):,.2f}" if metrics.get('sma20') else "n/a",
            f"{metrics.get('sma50', float('nan')):,.2f}" if metrics.get('sma50') else "n/a",
            f"{metrics.get('drawdown')*100:.1f}%" if metrics.get('drawdown') is not None else "n/a",
            action,
        )
    console.print(dip_tbl)

    hold_tbl = Table(title="Holdings Snapshot")
    hold_tbl.add_column("Ticker", style="cyan")
    hold_tbl.add_column("Qty", justify="right")
    hold_tbl.add_column("Avg ₹", justify="right")
    hold_tbl.add_column("LTP ₹", justify="right")
    hold_tbl.add_column("Value ₹", justify="right")
    hold_tbl.add_column("P&L ₹", justify="right")
    hold_tbl.add_column("P&L %", justify="right")

    for plan in plans:
        val = plan.valuation
        pnl_color = "green" if val["pnl"] >= 0 else "red"
        pct = val["pnl_pct"] * 100 if val["pnl_pct"] is not None else None
        pct_color = "green" if pct is not None and pct >= 0 else "red"
        hold_tbl.add_row(
            plan.ticker,
            f"{plan.quantity:,.0f}",
            f"{plan.avg_price:,.2f}",
            f"{plan.latest_price:,.2f}",
            f"{val['market_value']:,.2f}",
            f"[{pnl_color}]{val['pnl']:,.2f}[/]",
            f"[{pct_color}]{pct:.2f}%[/]" if pct is not None else "n/a",
        )
    console.print(hold_tbl)

    summary_tbl = Table(title="Summary")
    summary_tbl.add_column("Bucket")
    summary_tbl.add_column("₹", justify="right")
    summary_tbl.add_column("Notes")
    summary_tbl.add_row("Core SIP", f"{totals['core_budget']:,.0f}", "Split evenly across 2 tranches")
    summary_tbl.add_row(
        "Dip pot", f"{totals['dip_budget']:,.0f}",
        f"Deploy now: ₹{totals['deploy_now']:,.0f} | Carry: ₹{totals['carry_forward']:,.0f}"
    )
    summary_tbl.add_row(
        "Buffer", f"{totals['buffer_budget']:,.0f}",
        f"{buffer_summary['buffer_action']} (₹{buffer_summary['buffer_to_deploy']:,.0f})"
    )
    summary_tbl.add_row(
        "Holdings", f"{totals['market_value']:,.0f}",
        f"Unrealised P&L: ₹{totals['unrealized_pnl']:,.0f}"
    )
    console.print(summary_tbl)

    if buffer_summary.get("nifty_mtd") is not None:
        console.print(f"NIFTY MTD change: {buffer_summary['nifty_mtd']*100:.2f}%")
    if notes:
        console.print(f"[dim]Notes: {notes}[/dim]")


def next_buy_dates() -> Tuple[date, date]:
    def adjust_weekday(d: date) -> date:
        if d.weekday() == 5:
            return d + timedelta(days=2)
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    today = date.today()
    if today.day <= 15:
        first = date(today.year, today.month, 4)
        second = date(today.year, today.month, 15)
    else:
        nxt = today + relativedelta(months=1)
        first = date(nxt.year, nxt.month, 4)
        second = date(nxt.year, nxt.month, 15)

    return adjust_weekday(first), adjust_weekday(second)


def persist_history(plan: List[TickerPlan], totals: dict, buffer_summary: dict, config: dict) -> Optional[dict]:
    history_path = Path(config.get("reporting", {}).get("history_file", "reports/sip_history.json"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if history_path.exists():
        try:
            previous = load_json(history_path)
        except Exception:
            previous = None
    snapshot = {
        "generated_at": datetime.utcnow().isoformat(),
        "plans": [
            {
                "ticker": p.ticker,
                "market_value": p.valuation["market_value"],
                "pnl": p.valuation["pnl"],
                "avg_price": p.avg_price,
                "quantity": p.quantity,
                "latest_price": p.latest_price,
            }
            for p in plan
        ],
        "totals": totals,
        "buffer": buffer_summary,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(snapshot), f, indent=2)
    return previous


def compute_deltas(current_totals: dict, previous_snapshot: Optional[dict]) -> Optional[dict]:
    if not previous_snapshot:
        return None
    prev_totals = previous_snapshot.get("totals", {})
    deltas = {}
    for key in ["market_value", "unrealized_pnl", "deploy_now", "carry_forward"]:
        if key in current_totals and key in prev_totals:
            deltas[key] = current_totals[key] - prev_totals.get(key, 0)
    return deltas


def save_reports(plan: List[TickerPlan], totals: dict, buffer_summary: dict, config: dict) -> List[str]:
    reporting = config.get("reporting", {})
    output_dir = Path(reporting.get("output_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    files = []

    df_rows = []
    for p in plan:
        df_rows.append({
            "Ticker": p.ticker,
            "Qty": p.quantity,
            "AvgPrice": p.avg_price,
            "LatestPrice": p.latest_price,
            "PrevMonthClose": p.prev_month_close,
            "DropPct": p.drop_pct,
            "DipAllocation": p.allocations["dip"],
            "CoreAllocation": p.allocations["core"],
            "MarketValue": p.valuation["market_value"],
            "UnrealizedPnL": p.valuation["pnl"],
            "SMA20": p.metrics.get("sma20"),
            "SMA50": p.metrics.get("sma50"),
            "Drawdown": p.metrics.get("drawdown"),
            "Vol20": p.metrics.get("vol20"),
        })
    df = pd.DataFrame(df_rows)

    if reporting.get("save_csv", True):
        csv_path = output_dir / f"sip_plan_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        files.append(str(csv_path))
    if reporting.get("save_markdown", True):
        md_path = output_dir / f"sip_plan_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# SIP Plan — {datetime.utcnow().isoformat()}\n\n")
            try:
                md_table = df.to_markdown(index=False)
            except ImportError:
                md_table = df.to_string(index=False)
            f.write(md_table)
            f.write("\n\n## Summary\n")
            summary_payload = {"totals": totals, "buffer": buffer_summary}
            f.write(json.dumps(make_json_safe(summary_payload), indent=2))
        files.append(str(md_path))
    return files


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"₹{value:,.0f}"


def build_email_body(plans: List[TickerPlan], totals: dict, buffer_summary: dict, actions: List[str]) -> Tuple[str, str]:
    subject = f"SIP Plan {date.today().isoformat()}"
    plain_lines = ["SIP Planner Summary", ""]
    if actions:
        plain_lines.append("Actions:")
        for act in actions:
            plain_lines.append(f"- {act}")
        plain_lines.append("")
    plain_lines.append(f"Core budget: {format_currency(totals['core_budget'])}")
    plain_lines.append(f"Dip budget: {format_currency(totals['dip_budget'])} (deploy {format_currency(totals['deploy_now'])})")
    plain_lines.append(f"Buffer: {buffer_summary['buffer_action']} (suggested {format_currency(buffer_summary['buffer_to_deploy'])})")
    plain_lines.append(f"Holdings MV: {format_currency(totals['market_value'])}")
    plain_lines.append(f"Unrealised P&L: {format_currency(totals['unrealized_pnl'])}")

    plain_lines.append("\nTickers:")
    for p in plans:
        plain_lines.append(
            f"- {p.ticker}: LTP ₹{p.latest_price:,.2f}, dip ₹{p.allocations['dip']:,.0f}, drop {p.drop_pct*100:.1f}%" if p.drop_pct is not None else f"- {p.ticker}: LTP ₹{p.latest_price:,.2f}, dip ₹{p.allocations['dip']:,.0f}"
        )

    plain_body = "\n".join(plain_lines)

    html_rows = []
    for p in plans:
        drop = f"{p.drop_pct*100:.1f}%" if p.drop_pct is not None else "n/a"
        dip_amt = f"₹{p.allocations['dip']:,.0f}"
        action = "BUY" if p.allocations["dip"] > 0 and p.dip_trigger else "HOLD"
        html_rows.append(
            f"<tr><td>{p.ticker}</td><td>{p.latest_price:,.2f}</td><td>{dip_amt}</td><td>{drop}</td><td>{action}</td></tr>"
        )
    html_body = f"""
    <html><body>
    <h2>SIP Planner Summary</h2>
    <p>Core budget: {format_currency(totals['core_budget'])}<br/>
       Dip budget: {format_currency(totals['dip_budget'])} (deploy {format_currency(totals['deploy_now'])})<br/>
       Buffer: {buffer_summary['buffer_action']} (₹{buffer_summary['buffer_to_deploy']:,.0f})<br/>
       Holdings MV: {format_currency(totals['market_value'])}<br/>
       Unrealised P&L: {format_currency(totals['unrealized_pnl'])}</p>
    {('<h3>Actions</h3><ul>' + ''.join(f'<li>{a}</li>' for a in actions) + '</ul>') if actions else ''}
    <table border="1" cellpadding="6" cellspacing="0">
        <tr><th>Ticker</th><th>LTP ₹</th><th>Dip ₹</th><th>Δ prev month</th><th>Action</th></tr>
        {''.join(html_rows)}
    </table>
    </body></html>
    """
    return subject, (plain_body, html_body)


def send_email(notification_cfg: dict, subject: str, bodies: Tuple[str, str]) -> None:
    email_cfg = notification_cfg.get("email", {})
    if not email_cfg.get("enabled", True):
        return
    sender = os.getenv("SIP_EMAIL_FROM") or email_cfg.get("from") or os.getenv("MAIL_FROM")
    recipient = os.getenv("SIP_EMAIL_TO") or email_cfg.get("to") or os.getenv("MAIL_TO") or sender
    smtp_user = os.getenv("SIP_SMTP_USER") or email_cfg.get("user") or sender
    smtp_pass = os.getenv("SIP_SMTP_PASS") or email_cfg.get("password") or os.getenv("SMTP_PASS")
    smtp_host = os.getenv("SIP_SMTP_SERVER") or email_cfg.get("host") or "smtp.gmail.com"
    smtp_port = int(os.getenv("SIP_SMTP_PORT") or email_cfg.get("port", 465))
    use_ssl = env_flag("SIP_SMTP_USE_SSL") or email_cfg.get("ssl", True)
    use_tls = env_flag("SIP_SMTP_USE_TLS") or email_cfg.get("tls", False)

    if not (sender and recipient and smtp_pass):
        console.print("[yellow]Email not sent: missing credentials.[/yellow]")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    plain_body, html_body = bodies
    msg.set_content(plain_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
                smtp.login(smtp_user, smtp_pass)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as smtp:
                if use_tls:
                    smtp.starttls()
                smtp.login(smtp_user, smtp_pass)
                smtp.send_message(msg)
        console.print(f"[green]Email sent to {recipient}[/green]")
    except Exception as e:
        console.print(f"[red]Email send failed: {e}[/red]")


def send_slack(notification_cfg: dict, text: str) -> None:
    slack_cfg = notification_cfg.get("slack", {})
    if not slack_cfg.get("enabled"):
        return
    webhook = os.getenv("SIP_SLACK_WEBHOOK") or slack_cfg.get("webhook_url")
    if not webhook:
        console.print("[yellow]Slack webhook missing; skipping[/yellow]")
        return
    try:
        resp = requests.post(webhook, json={"text": text}, timeout=10)
        if resp.ok:
            console.print("[green]Slack notification sent[/green]")
        else:
            console.print(f"[red]Slack error {resp.status_code}: {resp.text[:120]}[/red]")
    except Exception as e:
        console.print(f"[red]Slack send failed: {e}[/red]")


def send_telegram(notification_cfg: dict, text: str) -> None:
    tg_cfg = notification_cfg.get("telegram", {})
    if not tg_cfg.get("enabled"):
        return
    token = os.getenv("SIP_TELEGRAM_TOKEN") or tg_cfg.get("bot_token")
    chat_id = os.getenv("SIP_TELEGRAM_CHAT_ID") or tg_cfg.get("chat_id")
    if not (token and chat_id):
        console.print("[yellow]Telegram credentials missing; skipping[/yellow]")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        if resp.ok:
            console.print("[green]Telegram notification sent[/green]")
        else:
            console.print(f"[red]Telegram error {resp.status_code}: {resp.text[:120]}[/red]")
    except Exception as e:
        console.print(f"[red]Telegram send failed: {e}[/red]")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIP planner and alerting tool")
    parser.add_argument("--config", help="Path to sip_config.json", default=None)
    parser.add_argument("--budget", type=int, help="Override monthly budget (₹)")
    parser.add_argument("--offline", action="store_true", help="Run using cached data only")
    parser.add_argument("--skip-email", action="store_true", help="Skip email notifications")
    parser.add_argument("--skip-slack", action="store_true", help="Skip Slack notifications")
    parser.add_argument("--skip-telegram", action="store_true", help="Skip Telegram notifications")
    parser.add_argument("--only", help="Comma-separated tickers to include")
    parser.add_argument("--notes", help="Optional note to print in output")
    parser.add_argument("--report-dir", help="Override reports output directory")
    parser.add_argument("--cache", help="Override cache path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.report_dir:
        config.setdefault("reporting", {})["output_dir"] = args.report_dir
    if args.cache:
        config.setdefault("data", {})["cache_path"] = args.cache

    offline = args.offline or config.get("data", {}).get("offline", False)
    cache_path = Path(config.get("data", {}).get("cache_path", DEFAULT_CACHE_PATH))
    fetcher = MarketDataFetcher(
        cache_path=cache_path,
        offline=offline,
        retries=int(config.get("data", {}).get("retry_attempts", 3)),
        backoff=float(config.get("data", {}).get("retry_backoff", 1.5)),
    )

    try:
        plans, totals, buffer_summary, actions = compute_plan(config, fetcher, args)
    except Exception as e:
        console.print(f"[red]Failed to compute plan: {e}[/red]")
        return 1

    notes = args.notes
    if not notes and DEFAULT_NOTES_PATH.exists():
        try:
            notes = DEFAULT_NOTES_PATH.read_text(encoding="utf-8").strip()
        except Exception:
            notes = None

    render_console(plans, totals, buffer_summary, notes)
    previous_snapshot = persist_history(plans, totals, buffer_summary, config)
    deltas = compute_deltas(totals, previous_snapshot)
    if deltas:
        console.print("[dim]Δ vs prior run:" + ", ".join(f" {k}: {v:+,.0f}" for k, v in deltas.items()) + "[/dim]")

    saved_files = save_reports(plans, totals, buffer_summary, config)
    if saved_files:
        console.print("Saved reports: " + ", ".join(saved_files))

    notifications_cfg = config.get("notifications", {})
    if args.skip_email:
        notifications_cfg.setdefault("email", {})["enabled"] = False
    if args.skip_slack:
        notifications_cfg.setdefault("slack", {})["enabled"] = False
    if args.skip_telegram:
        notifications_cfg.setdefault("telegram", {})["enabled"] = False

    subject, bodies = build_email_body(plans, totals, buffer_summary, actions)
    if notifications_cfg.get("email", {}).get("enabled", True):
        send_email(notifications_cfg, subject, bodies)
    slack_text = bodies[0].split("\n\n", 1)[0] + "\n" + "\n".join(actions)
    send_slack(notifications_cfg, slack_text)
    send_telegram(notifications_cfg, slack_text)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
