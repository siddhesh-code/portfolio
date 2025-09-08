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


# --- small significance thresholds (business-logic only; no API changes) ---
_EPS = 1e-9
_MACD_MIN_DIFF = 0.02       # minimal MACD-vs-signal separation to count a cross
_HIST_MIN_SLOPE = 0.01      # minimal histogram slope magnitude
_RSI_SLOPE_BENEFIT = 2
_BB_REENTRY_TOL = 1e-6      # tolerance to avoid equality flips
_VOL_SPIKE_RATIO = 1.5
_SR_PROX_PCT = 0.03         # within 3% of S/R
_SMA_NEUTRAL_BAND = 0.001   # 0.1% neutral zone to avoid tiny MA stack signals
_MIN_BODY_FRAC = 0.1        # ignore candle patterns on ultra tiny bodies (of total range)
_MIN_RANGE_FRAC = 0.002     # ignore patterns if total range < 0.2% of price (micro-bars)
_NEAR_ZERO_SCORE = 0.5      # treat |score| < this as neutral for ranking


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


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _last(df, col, n=1, default=0.0):
    try:
        v = float(df[col].iloc[-n])
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _hammer_like(o: float, h: float, l: float, c: float) -> bool:
    body = abs(c - o)
    total = max(h - l, _EPS)
    # ignore micro-bars (too noisy)
    if (total / max(abs(c), 1.0)) < _MIN_RANGE_FRAC or (body / total) < _MIN_BODY_FRAC:
        return False
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick / total) > 0.5 and (body / total) < 0.3 and (upper_wick / total) < 0.2


def _shooting_star(o: float, h: float, l: float, c: float) -> bool:
    body = abs(c - o)
    total = max(h - l, _EPS)
    if (total / max(abs(c), 1.0)) < _MIN_RANGE_FRAC or (body / total) < _MIN_BODY_FRAC:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick / total) > 0.5 and (body / total) < 0.3 and (lower_wick / total) < 0.2


def _engulfing(prev_o, prev_c, o, c) -> Tuple[bool, bool]:
    # returns (bullish_engulfing, bearish_engulfing)
    bull = (prev_c < prev_o) and (c > o) and (c >= prev_o) and (o <= prev_c)
    bear = (prev_c > prev_o) and (c < o) and (c <= prev_o) and (o >= prev_c)
    return bull, bear


def _morning_evening_star(df) -> Tuple[bool, bool]:
    # Heuristic 3-candle pattern with tiny-range guard
    try:
        o1, c1 = float(df['Open'].iloc[-3]), float(df['Close'].iloc[-3])
        o2, c2 = float(df['Open'].iloc[-2]), float(df['Close'].iloc[-2])
        o3, c3 = float(df['Open'].iloc[-1]), float(df['Close'].iloc[-1])
        h1, l1 = float(df['High'].iloc[-3]), float(df['Low'].iloc[-3])
        total1 = max(h1 - l1, _EPS)
        if (total1 / max(abs(c1), 1.0)) < _MIN_RANGE_FRAC:
            return False, False

        # Morning star: long down day, small body (gap/indecision), strong up day closing > midpoint of day1
        long_down = (c1 < o1) and abs(c1 - o1) > abs(o1) * 0.005
        indecision = abs(c2 - o2) < abs(c1 - o1) * 0.5
        strong_up = (c3 > o3) and (c3 > (o1 + c1) / 2)
        morning = bool(long_down and indecision and strong_up)

        # Evening star: mirror
        long_up = (c1 > o1) and abs(c1 - o1) > abs(o1) * 0.005
        indecision2 = abs(c2 - o2) < abs(c1 - o1) * 0.5
        strong_down = (c3 < o3) and (c3 < (o1 + c1) / 2)
        evening = bool(long_up and indecision2 and strong_down)
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
    if rsi < 30: 
        score += 6; reasons.append('RSI oversold')
    elif rsi > 70: 
        score -= 6; reasons.append('RSI overbought')
    # slope (light)
    if _is_finite(rsi_prev) and _is_finite(rsi) and rsi_prev != 0:
        if rsi_prev < rsi: score += _RSI_SLOPE_BENEFIT
        elif rsi_prev > rsi: score -= _RSI_SLOPE_BENEFIT

    # 2) MACD cross + histogram slope (with significance filters)
    crossed_up = (macd > macd_s) and (macd_prev <= macd_s_prev) and (abs(macd - macd_s) >= _MACD_MIN_DIFF)
    crossed_dn = (macd < macd_s) and (macd_prev >= macd_s_prev) and (abs(macd - macd_s) >= _MACD_MIN_DIFF)
    if crossed_up: 
        score += 6; reasons.append('MACD bull cross')
    if crossed_dn: 
        score -= 6; reasons.append('MACD bear cross')

    hist_slope = hist - hist_prev
    if abs(hist_slope) >= _HIST_MIN_SLOPE:
        if hist_slope > 0: score += 3
        if hist_slope < 0: score -= 3

    # 3) Bollinger pierce and close back inside (guard tiny flips)
    if (prev_close < bb_l - _BB_REENTRY_TOL) and (c > bb_l + _BB_REENTRY_TOL):
        score += 5; reasons.append('Re-entered above lower band')
    if (prev_close > bb_u + _BB_REENTRY_TOL) and (c < bb_u - _BB_REENTRY_TOL):
        score -= 5; reasons.append('Re-entered below upper band')

    # 4) Candle patterns (ignore micro-bars)
    if _hammer_like(o, h, l, c): 
        score += 5; reasons.append('Hammer-like')
    if _shooting_star(o, h, l, c): 
        score -= 5; reasons.append('Shooting star-like')
    bull_eng, bear_eng = _engulfing(po, pc, o, c)
    if bull_eng: 
        score += 5; reasons.append('Bullish engulfing')
    if bear_eng: 
        score -= 5; reasons.append('Bearish engulfing')
    ms, es = _morning_evening_star(df)
    if ms: 
        score += 6; reasons.append('Morning star')
    if es: 
        score -= 6; reasons.append('Evening star')

    # 5) Volume spike (ratio-based, avoid truthiness bug)
    if _is_finite(vol_sma) and vol_sma > 0 and _is_finite(vol) and (vol / vol_sma) > _VOL_SPIKE_RATIO:
        score += 4; reasons.append('Volume spike')

    # 6) Proximity to S/R (within SR_PROX_PCT)
    if _is_finite(s1) and s1 > 0 and _is_finite(close) and 0 < (close - s1) / max(close, _EPS) < _SR_PROX_PCT:
        score += 3; reasons.append('Near S1')
    if _is_finite(r1) and r1 > 0 and _is_finite(close) and 0 < (r1 - close) / max(close, _EPS) < _SR_PROX_PCT:
        score -= 3; reasons.append('Near R1')

    # 7) 5-day momentum flip
    try:
        m5 = _pct(df['Close'].iloc[-1], df['Close'].iloc[-6])
        m5_prev = _pct(df['Close'].iloc[-2], df['Close'].iloc[-7])
        if m5 > 0 and m5_prev < 0: 
            score += 4; reasons.append('Momentum flip up (5d)')
        if m5 < 0 and m5_prev > 0: 
            score -= 4; reasons.append('Momentum flip down (5d)')
    except Exception:
        pass

    # 8) SMA bounce/cross risk with neutral band
    if _is_finite(close) and _is_finite(sma20) and _is_finite(sma50):
        # neutral: if sma20 and sma50 are almost equal relative to price
        if abs(sma20 - sma50) / max(abs(close), 1.0) > _SMA_NEUTRAL_BAND:
            if close > sma20 > sma50: 
                score += 3
            if close < sma20 < sma50: 
                score -= 3

    # 9) ADI slope if present (5-bar change)
    try:
        adi = float(df['ADI'].iloc[-1]) - float(df['ADI'].iloc[-6])
        if math.isfinite(adi):
            if adi > 0: score += 2
            if adi < 0: score -= 2
    except Exception:
        pass

    # 10) Gap and fill
    try:
        today_gap = o - pc
        if today_gap < 0 and c > o: 
            score += 2  # gap down then reclaim
        if today_gap > 0 and c < o: 
            score -= 2  # gap up then fade
    except Exception:
        pass

    features = {
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd_s,
        'hist_slope': hist_slope,
        'bb_lower': bb_l,
        'bb_upper': bb_u,
        'volume_spike': (_is_finite(vol_sma) and vol_sma > 0 and _is_finite(vol) and (vol / vol_sma) > _VOL_SPIKE_RATIO),
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
            closes = [float(x) if math.isfinite(float(x)) else None for x in df['Close'].tail(30).tolist()]
            closes = [x for x in closes if x is not None]
        except Exception:
            closes = []

        # treat tiny |score| as neutral to avoid noise
        abs_score = abs(score) if abs(score) >= _NEAR_ZERO_SCORE else 0.0
        direction = 'Bullish' if score > 0 and abs_score > 0 else 'Bearish' if score < 0 and abs_score > 0 else 'Neutral'

        item = {
            'symbol': sym,
            'name': base.get('name') or sym.split('.')[0],
            'price': base.get('price') or float(df['Close'].iloc[-1]),
            'score': round(abs_score, 2),
            'direction': direction,
            'reasons': reasons,
            'indicators': base.get('indicators', {}),
            'link': f"/analyze?symbols={sym}",
            'spark': closes,
        }
        if direction == 'Bullish':
            out_bull.append(item)
        elif direction == 'Bearish':
            out_bear.append(item)
        # neutrals are intentionally dropped

    out_bull.sort(key=lambda x: x['score'], reverse=True)
    out_bear.sort(key=lambda x: x['score'], reverse=True)
    return { 'bullish': out_bull[:top_n], 'bearish': out_bear[:top_n] }
