"""
Data fetching module for stock data
"""

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random
from config import NIFTY50_SYMBOLS, NIFTY500_SYMBOLS
import sys
from indicators import calculate_technical_indicators


# ---- Internal tunables (business logic only; public API unchanged) ----
_FETCH_PERIOD = "1y"          # keep as requested
_FETCH_INTERVAL = "1d"
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.6           # seconds
_BACKOFF_JITTER = (0.0, 0.4)  # random jitter window
_MIN_ROWS_FOR_TA = 60         # minimal bars before we attempt indicators
_MAX_WORKERS_CAP = 32         # safety cap for threads


def _sleep_with_backoff(attempt: int) -> None:
    """Exponential backoff with jitter."""
    delay = (_BACKOFF_BASE * (2 ** attempt)) + random.uniform(*_BACKOFF_JITTER)
    time.sleep(delay)


def _normalize_history(df):
    """Ensure history has expected columns and tz-naive index; auto-adjusted OHLCV."""
    try:
        if df is None or df.empty:
            return df
        # Drop multiindex columns if present (e.g., from group downloads)
        if hasattr(df.columns, "levels") and len(getattr(df.columns, "levels", [])) > 1:
            # Not used here; single-ticker Ticker.history doesn't return multiindex
            df = df.copy()
        # Normalize tz
        try:
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert(None)
        except Exception:
            try:
                df.index = df.index.tz_localize(None)
            except Exception:
                pass
        # Ensure expected cols exist
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in needed):
            return None
        # Cast to float where appropriate
        for c in ["Open", "High", "Low", "Close"]:
            try:
                df[c] = df[c].astype(float)
            except Exception:
                return None
        try:
            df["Volume"] = df["Volume"].astype(float)
        except Exception:
            df["Volume"] = 0.0
        # Drop rows with obviously bad prices
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return None


def _safe_get_info(ticker: yf.Ticker) -> dict:
    """Robust info retrieval: try get_info(), fallback to fast_info snapshot."""
    info = {}
    try:
        # yfinance >=0.2.28: get_info() is preferred over .info property
        info = ticker.get_info() or {}
    except Exception:
        pass
    if not info:
        try:
            fi = getattr(ticker, "fast_info", None)
            if fi:
                # Map a few helpful fields
                info = {
                    "shortName": getattr(fi, "short_name", None) or getattr(fi, "symbol", None),
                    "longName": getattr(fi, "short_name", None),
                    "currency": getattr(fi, "currency", None),
                    "exchange": getattr(fi, "exchange", None),
                    "marketCap": getattr(fi, "market_cap", None),
                }
        except Exception:
            info = {}
    return info


def fetch_stock_data(symbol):
    """Fetch data for a single stock symbol"""
    try:
        # tiny stagger to reduce thundering herd
        time.sleep(random.uniform(0.0, 0.05))

        ticker = yf.Ticker(symbol)

        hist = None
        last_err = None
        for attempt in range(_MAX_RETRIES):
            try:
                # Use auto_adjust=True to incorporate splits/divs in OHLC
                hist = ticker.history(
                    period=_FETCH_PERIOD,
                    interval=_FETCH_INTERVAL,
                    auto_adjust=True,
                    actions=False,
                    prepost=False,
                    back_adjust=False,
                    repair=True,  # yfinance can fix some missing rows
                    rounding=False,
                )
                if hist is not None and not hist.empty:
                    break
            except Exception as e:
                last_err = e
            _sleep_with_backoff(attempt)

        if hist is None or hist.empty:
            print(f"No historical data available for {symbol}")
            return symbol, None

        hist = _normalize_history(hist)
        if hist is None or hist.empty:
            print(f"Invalid/insufficient history for {symbol}")
            return symbol, None

        # Guard: ensure we have enough bars for indicators (still return minimal payload if not)
        info = _safe_get_info(ticker)

        technical = None
        if len(hist) >= _MIN_ROWS_FOR_TA:
            try:
                technical = calculate_technical_indicators(hist)
            except Exception as e:
                # If indicators fail, we still return history/info
                print(f"Indicators failed for {symbol}: {str(e)}")
                technical = None

        data = {
            'info': info,
            'history': hist,
            'technical': technical,
        }
        return symbol, data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return symbol, None


def _fetch_many(symbols, max_workers=5, desc="Fetching stock data"):
    """Shared concurrent fetcher with progress bar and robust result handling."""
    data = {}
    symbols = list(symbols or [])
    if not symbols:
        return data

    # Respect caller's max_workers but cap overall
    try:
        max_workers_eff = max(1, min(int(max_workers), _MAX_WORKERS_CAP, len(symbols)))
    except Exception:
        max_workers_eff = max(1, min(_MAX_WORKERS_CAP, len(symbols)))

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers_eff) as executor:
        futures = {executor.submit(fetch_stock_data, sym): sym for sym in symbols}
        iterator = as_completed(futures)

        # tqdm: only show if stderr is a tty
        progress = tqdm(
            total=len(symbols),
            desc=desc,
            disable=not sys.stderr.isatty()
        )

        for fut in iterator:
            sym = futures[fut]
            try:
                symbol, result = fut.result()
                if result is not None:
                    data[symbol] = result
            except Exception as e:
                print(f"Worker failed for {sym}: {str(e)}")
            finally:
                progress.update(1)

        progress.close()

    end_time = time.time()
    print(f"\nFetched data for {len(data)} stocks in {end_time - start_time:.2f} seconds")
    return data


def get_nifty50_data(symbols=None, max_workers=5):
    """Fetch data for all Nifty 50 stocks in parallel"""
    if symbols is None:
        symbols = NIFTY50_SYMBOLS
    return _fetch_many(symbols, max_workers=max_workers, desc="Fetching NIFTY50 data")



def get_nifty500_data(symbols=None, max_workers=5):
    """Fetch data for all Nifty 500 stocks in parallel"""
    if symbols is None:
        symbols = NIFTY500_SYMBOLS
    # print stock whose data was not able to fetch
    return _fetch_many(symbols, max_workers=max_workers, desc="Fetching NIFTY500 data")


