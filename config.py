"""
Configuration and constants for the portfolio analysis
"""

# List of Nifty 50 stock symbols (NSE format: SYMBOL.NS)
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS",
    "BHARTIARTL.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS",
    "AXISBANK.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "TITAN.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "WIPRO.NS", "TECHM.NS", "DIVISLAB.NS", "GRASIM.NS",
    "CIPLA.NS", "ADANIPORTS.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS", "ONGC.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "BPCL.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "DRREDDY.NS", "COALINDIA.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS", "M&M.NS",
    "SBILIFE.NS", "INDUSINDBK.NS", "SHREECEM.NS", "NTPC.NS",
    "BAJAJ-AUTO.NS", "ADANIENT.NS", "UPL.NS", "TATAMOTORS.NS", "TATACONSUM.NS"
]

# Technical Analysis Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
VOLUME_PERIOD = 20
VOLATILITY_PERIOD = 20
SMA_PERIODS = [20, 50]
EMA_PERIOD = 20

# Example portfolio positions for KPI computation
# Replace with your actual holdings or wire to a DB later
POSITIONS = [
    {"symbol": "RELIANCE.NS", "qty": 10, "avg_price": 2500.0},
    {"symbol": "HDFCBANK.NS", "qty": 15, "avg_price": 1550.0},
    {"symbol": "ICICIBANK.NS", "qty": 12, "avg_price": 950.0}
]

# -----------------------------
# Universes
# -----------------------------

def _normalize_ns(symbols):
    """Normalize a list of tickers to uppercase NSE format (SYMBOL.NS), deduped."""
    out = []
    seen = set()
    for s in symbols or []:
        if not isinstance(s, str) or not s.strip():
            continue
        sym = s.strip().upper()
        if not sym.endswith('.NS'):
            sym = sym + '.NS'
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out

# Curated starter set for the NIFTY 500 universe.
# Note: This is intentionally concise and normalized; extend as needed.
_NIFTY500_STARTER = [
    # Large Caps (beyond NIFTY50) – examples
    'ADANIGREEN', 'ADANIGAS', 'DMART', 'BAJAJHLDNG', 'GODREJCP', 'PIDILITIND',
    'BANKBARODA', 'PNB', 'CANBK', 'IDFCFIRSTB', 'BANDHANBNK', 'YESBANK',
    'DLF', 'HAVELLS', 'ABB', 'SIEMENS', 'MUTHOOTFIN', 'CHOLAFIN', 'AUROPHARMA',
    'ICICIGI', 'ICICIPRULI', 'MUTHOOTFIN', 'PEL', 'LUPIN', 'TORNTPHARM',
    # Popular midcaps
    'IRCTC', 'HAL', 'POLYCAB', 'ZYDUSLIFE', 'TATAELXSI', 'AIAENG', 'SRF',
    'PAGEIND', 'MPHASIS', 'PERSISTENT', 'INDIAMART', 'BATAINDIA', 'TRENT',
    'TVSMOTOR', 'BHARATFORG', 'ABBOTINDIA', 'GLAND', 'GUJGASLTD', 'ICICISIGI',
    # Add more as needed ...
]

# Final NIFTY500 list (normalized, deduplicated). If you maintain a complete list,
# replace `_NIFTY500_STARTER` with your full set or load from a file.
NIFTY500_SYMBOLS = _normalize_ns(NIFTY50_SYMBOLS + _NIFTY500_STARTER)
