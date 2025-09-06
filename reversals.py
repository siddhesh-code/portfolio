"""
Reversal screener inspired by established technical literature.

Detects bullish/bearish reversal candidates using 10+ signals:
 - RSI level and 3-day slope
 - MACD signal cross and histogram slope
 - Bollinger position (pierce/close back inside)
 - Candlestick patterns: hammer/shooting star, bullish/bearish engulfing,
   morning/evening star (heuristic, 3-candle)
 - Volume spike vs 20d SMA
 - Proximity to support/resistance (reversion zones)
 - 5-day momentum flip (neg→pos or pos→neg)
 - SMA(20) vs SMA(50) inflection (bounce/cross risk)
 - Accumulation/Distribution (ADI) slope (optional)

Produces a score and reasons per symbol. Positive score => bullish, negative => bearish.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math


def _pct(a: float, b: float) -> float:
    try:
        if b == 0:
            return 0.0
        return (a - b) / b * 100.0
    except Exception:
        return 0.0


def _slope(x2: float, x1: float) -> float:
    try:
        return x2 - x1
    except Exception:
        return 0.0


def _last(df, col, n=1, default=0.0):
    try:
        return float(df[col].iloc[-n])
    except Exception:
        return default


def _hammer_like(o: float, h: float, l: float, c: float) -> bool:
    body = abs(c - o)
    total = max(h - l, 1e-9)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick / total) > 0.5 and (body / total) < 0.3 and upper_wick / total < 0.2


def _shooting_star(o: float, h: float, l: float, c: float) -> bool:
    body = abs(c - o)
    total = max(h - l, 1e-9)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick / total) > 0.5 and (body / total) < 0.3 and lower_wick / total < 0.2


def _engulfing(prev_o, prev_c, o, c) -> Tuple[bool, bool]:
    # returns (bullish_engulfing, bearish_engulfing)
    bull = (prev_c < prev_o) and (c > o) and (c >= prev_o) and (o <= prev_c)
    bear = (prev_c > prev_o) and (c < o) and (c <= prev_o) and (o >= prev_c)
    return bull, bear


def _morning_evening_star(df) -> Tuple[bool, bool]:
    # Heuristic 3-candle pattern
    try:
        o1, c1 = float(df['Open'].iloc[-3]), float(df['Close'].iloc[-3])
        o2, c2 = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])
        o3, c3 = float(df['Open'].iloc[-1]), float(df['Close'].iloc[-1])
        # Morning star: long down day, small body (gap or indecision), strong up day closing > midpoint of day1
        long_down = (c1 < o1) and abs(c1 - o1) > abs(o1) * 0.005
        indecision = abs(c2 - o2) < abs(c1 - o1) * 0.5
        strong_up = (c3 > o3) and (c3 > (o1 + c1) / 2)
        morning = long_down and indecision and strong_up
        # Evening star: mirror
        long_up = (c1 > o1) and abs(c1 - o1) > abs(o1) * 0.005
        indecision2 = abs(c2 - o2) < abs(c1 - o1) * 0.5
        strong_down = (c3 < o3) and (c3 < (o1 + c1) / 2)
        evening = long_up and indecision2 and strong_down
        return morning, evening
    except Exception:
        return False, False


def score_reversal(df) -> Tuple[float, List[str], Dict[str, Any]]:
    """Return (score, reasons, features). Positive => bullish, negative => bearish."""
    rsi = _last(df, 'RSI')
    rsi_prev = _last(df, 'RSI', 2)
    macd = _last(df, 'MACD')
    macd_s = _last(df, 'MACD_signal')
    macd_prev = _last(df, 'MACD', 2)
    macd_s_prev = _last(df, 'MACD_signal', 2)
    hist = macd - macd_s
    hist_prev = macd_prev - macd_s_prev
    bb_u = _last(df, 'BB_upper')
    bb_l = _last(df, 'BB_lower')
    close = _last(df, 'Close')
    prev_close = _last(df, 'Close', 2)
    sma20 = _last(df, 'SMA_20')
    sma50 = _last(df, 'SMA_50')
    vol = _last(df, 'Volume')
    vol_sma = _last(df, 'Volume_SMA')
    s1 = _last(df, 'Support_1')
    r1 = _last(df, 'Resistance_1')

    # Candlestick of last two bars
    try:
        o, h, l, c = float(df['Open'].iloc[-1]), float(df['High'].iloc[-1]), float(df['Low'].iloc[-1]), float(df['Close'].iloc[-1])
        po, ph, pl, pc = float(df['Open'].iloc[-2]), float(df['High'].iloc[-2]), float(df['Low'].iloc[-2]), float(df['Close'].iloc[-2])
    except Exception:
        o = h = l = c = po = ph = pl = pc = close

    score = 0.0
    reasons: List[str] = []

    # 1) RSI extremes + slope
    if rsi < 30: score += 6; reasons.append('RSI oversold')
    if rsi > 70: score -= 6; reasons.append('RSI overbought')
    if rsi_prev < rsi: score += 2
    if rsi_prev > rsi: score -= 2

    # 2) MACD cross + histogram slope
    if macd > macd_s and macd_prev <= macd_s_prev: score += 6; reasons.append('MACD bull cross')
    if macd < macd_s and macd_prev >= macd_s_prev: score -= 6; reasons.append('MACD bear cross')
    if hist > hist_prev: score += 3
    if hist < hist_prev: score -= 3

    # 3) Bollinger pierce and close back inside
    if c > bb_l and prev_close < bb_l: score += 5; reasons.append('Re-entered above lower band')
    if c < bb_u and prev_close > bb_u: score -= 5; reasons.append('Re-entered below upper band')

    # 4) Candle patterns
    if _hammer_like(o, h, l, c): score += 5; reasons.append('Hammer-like')
    if _shooting_star(o, h, l, c): score -= 5; reasons.append('Shooting star-like')
    bull_eng, bear_eng = _engulfing(po, pc, o, c)
    if bull_eng: score += 5; reasons.append('Bullish engulfing')
    if bear_eng: score -= 5; reasons.append('Bearish engulfing')
    ms, es = _morning_evening_star(df)
    if ms: score += 6; reasons.append('Morning star')
    if es: score -= 6; reasons.append('Evening star')

    # 5) Volume spike
    if vol_sma and vol > 1.5 * vol_sma: score += 4; reasons.append('Volume spike')

    # 6) Proximity to S/R
    if s1 and 0 < (close - s1) / close < 0.03: score += 3; reasons.append('Near S1')
    if r1 and 0 < (r1 - close) / close < 0.03: score -= 3; reasons.append('Near R1')

    # 7) 5-day momentum flip
    try:
        m5 = _pct(df['Close'].iloc[-1], df['Close'].iloc[-6])
        m5_prev = _pct(df['Close'].iloc[-2], df['Close'].iloc[-7])
        if m5 > 0 and m5_prev < 0: score += 4; reasons.append('Momentum flip up (5d)')
        if m5 < 0 and m5_prev > 0: score -= 4; reasons.append('Momentum flip down (5d)')
    except Exception:
        pass

    # 8) SMA bounce/cross risk
    if close > sma20 > sma50: score += 3
    if close < sma20 < sma50: score -= 3

    # 9) ADI slope if present
    try:
        adi = df['ADI'].iloc[-1] - df['ADI'].iloc[-6]
        if adi > 0: score += 2
        if adi < 0: score -= 2
    except Exception:
        pass

    # 10) Gap and fill
    try:
        prev_gap = po - pc
        today_gap = o - pc
        if today_gap < 0 and c > o: score += 2  # gap down then reclaim
        if today_gap > 0 and c < o: score -= 2  # gap up then fade
    except Exception:
        pass

    features = {
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd_s,
        'hist_slope': hist - hist_prev,
        'bb_lower': bb_l,
        'bb_upper': bb_u,
        'volume_spike': (vol_sma and vol > 1.5 * vol_sma),
        'hammer': _hammer_like(o, h, l, c),
        'shooting_star': _shooting_star(o, h, l, c),
        'bull_engulfing': bull_eng,
        'bear_engulfing': bear_eng,
    }

    return score, reasons, features


def rank_reversals(results: Dict[str, Dict[str, Any]], formatted: List[Dict[str, Any]], top_n: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    """Return {'bullish': [...], 'bearish': [...]} lists ranked by absolute score."""
    out_bull, out_bear = [], []
    form_map = {s['symbol']: s for s in formatted}
    for sym, data in results.items():
        df = data.get('technical')
        if df is None or len(df) < 30:
            continue
        score, reasons, feats = score_reversal(df)
        base = form_map.get(sym, {})
        # Build sparkline (close-only, last 30)
        try:
            closes = [float(x) for x in df['Close'].tail(30).tolist()]
        except Exception:
            closes = []

        item = {
            'symbol': sym,
            'name': base.get('name') or sym.split('.')[0],
            'price': base.get('price') or float(df['Close'].iloc[-1]),
            'score': round(abs(score), 2),
            'direction': 'Bullish' if score > 0 else 'Bearish' if score < 0 else 'Neutral',
            'reasons': reasons,
            'indicators': base.get('indicators', {}),
            'link': f"/analyze?symbols={sym}",
            'spark': closes,
        }
        if score > 0:
            out_bull.append(item)
        elif score < 0:
            out_bear.append(item)
    out_bull.sort(key=lambda x: x['score'], reverse=True)
    out_bear.sort(key=lambda x: x['score'], reverse=True)
    return { 'bullish': out_bull[:top_n], 'bearish': out_bear[:top_n] }
