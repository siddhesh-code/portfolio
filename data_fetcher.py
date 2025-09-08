"""
Data fetching module for stock data (NSE) with caching and bulk fetch support.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

from config import NIFTY50_SYMBOLS
from indicators import calculate_technical_indicators


logger = logging.getLogger(__name__)

# Simple in-memory TTL cache for fetched/processed symbols
_CACHE: Dict[Tuple[str, ...], Tuple[float, Dict[str, dict]]] = {}
_CACHE_TTL_SECONDS = 300.0  # 5 minutes


def _norm_symbols(symbols: List[str] | None) -> List[str]:
    if not symbols:
        return []
    out = []
    seen = set()
    for s in symbols:
        if not isinstance(s, str) or not s.strip():
            continue
        sym = s.strip().upper()
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _bulk_fetch_histories(symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, object]:
    """Fetch OHLCV histories for multiple symbols using a single yahoo call.

    Returns a mapping symbol -> DataFrame.
    """
    if not symbols:
        return {}
    try:
        df = yf.download(symbols, period=period, interval=interval, group_by="ticker", threads=True, auto_adjust=False)
    except Exception as e:
        logger.warning("yf.download failed (%s). Falling back to per-symbol fetch.", e)
        return {}

    out: Dict[str, object] = {}
    try:
        # Multi-symbol => columns MultiIndex (symbol, field)
        if hasattr(df, "columns") and getattr(df.columns, "levels", None) and len(df.columns.levels) >= 2:
            for sym in symbols:
                try:
                    part = df[sym]
                    if getattr(part, "empty", True):
                        continue
                    out[sym] = part
                except Exception:
                    continue
        else:
            # Single symbol => standard columns
            if not getattr(df, "empty", True):
                out[symbols[0]] = df
    except Exception as e:
        logger.debug("Error splitting bulk df: %s", e)
    return out


def _compute_symbol_payload(sym: str, hist) -> Tuple[str, dict | None]:
    """Compute the per-symbol payload (history + technical)."""
    try:
        if hist is None or getattr(hist, "empty", True):
            return sym, None
        # Trim to last ~260 rows to speed up indicators while supporting 200-SMA
        tail = hist.tail(260)
        tech = calculate_technical_indicators(tail)
        return sym, {
            'info': {},  # avoid slow/brittle ticker.info
            'history': tail,
            'technical': tech,
        }
    except Exception as e:
        logger.debug("Error computing technicals for %s: %s", sym, e)
        return sym, None


def fetch_stock_data(symbol: str) -> Tuple[str, dict | None]:
    """Fetch data for a single stock symbol (fallback path)."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        return _compute_symbol_payload(symbol, hist)
    except Exception as e:
        logger.debug("Error fetching %s: %s", symbol, e)
        return symbol, None


def get_nifty50_data(symbols: List[str] | None = None, max_workers: int = 5) -> Dict[str, dict]:
    """Fetch data for provided symbols (default NIFTY50) with caching and bulk fetch.

    Returns mapping symbol -> {'info': {}, 'history': df, 'technical': df}
    """
    start = time.time()
    if symbols is None:
        symbols = NIFTY50_SYMBOLS
    symbols = _norm_symbols(symbols)

    if not symbols:
        return {}

    # Cache check
    key = tuple(sorted(symbols))
    now = time.time()
    cached = _CACHE.get(key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        logger.debug("Cache hit for %d symbols", len(symbols))
        return cached[1]

    # Try bulk fetch first
    bulk = _bulk_fetch_histories(symbols)
    results: Dict[str, dict] = {}
    if bulk:
        for sym in symbols:
            sym_hist = bulk.get(sym)
            sym_key, payload = _compute_symbol_payload(sym, sym_hist)
            if payload is not None:
                results[sym_key] = payload
    else:
        # Fallback: per-symbol with threads, respecting caller's max_workers
        pool_size = min(max(1, int(max_workers)), len(symbols), 32)
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for sym, payload in executor.map(fetch_stock_data, symbols):
                if payload is not None:
                    results[sym] = payload

    # Update cache
    _CACHE[key] = (now, results)
    logger.debug("Fetched %d/%d symbols in %.2fs", len(results), len(symbols), time.time() - start)
    return results
