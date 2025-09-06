"""
AI-Powered Analysis utilities.

Implements the Conviction Gate described in docs/ARCHITECTURE.md
to rank and filter stocks for high-probability trades.

The gate computes a 0–100 conviction score per stock using
technical context (trend, RSI, MACD, SMA), risk/reward, health,
volume and volatility, then filters by a minimum threshold.

Usage:
    from ai_analysis import conviction_gate
    filtered = conviction_gate(stocks, min_score=70)

Where `stocks` is the list of dictionaries returned by
`stock_analyzer.format_stock_analysis`.
"""
from __future__ import annotations
from typing import List, Dict, Any


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    try:
        for key in path:
            cur = cur.get(key, {}) if isinstance(cur, dict) else {}
        return cur if cur is not None else default
    except Exception:
        return default


def conviction_gate(
    stocks: List[Dict[str, Any]],
    min_score: int = 70,
    prefer_action: str | None = None,
) -> List[Dict[str, Any]]:
    """Score and filter stocks by conviction.

    - min_score: Minimum score (0–100) to pass the gate.
    - prefer_action: Optional bias toward a specific action ('Buy', 'Sell').

    Returns the list of stocks annotated with `conviction` and filtered by threshold,
    sorted by score descending.
    """
    scored: List[Dict[str, Any]] = []

    for stock in stocks or []:
        reasons: List[str] = []
        score = 0.0

        # Health score (0–100) up to 30 pts
        health = float(stock.get('health_score', 0))
        score += 0.30 * max(0.0, min(100.0, health))
        reasons.append(f"Health {int(health)} contributes {0.30*min(100, max(0, int(health))):.0f}")

        # Risk/Reward weighting (encourage >2)
        rr = float(stock.get('risk_reward', 0) or 0)
        if rr >= 2.5:
            score += 18; reasons.append("Risk/Reward ≥2.5 +18")
        elif rr >= 2.0:
            score += 14; reasons.append("Risk/Reward ≥2.0 +14")
        elif rr >= 1.5:
            score += 8; reasons.append("Risk/Reward ≥1.5 +8")
        elif rr >= 1.0:
            score += 3; reasons.append("Risk/Reward ≥1.0 +3")
        else:
            score -= 12; reasons.append("Risk/Reward <1.0 −12")

        # Trend direction and strength
        trend = str(stock.get('trend', 'Sideways'))
        if trend == 'Uptrend':
            score += 12; reasons.append("Uptrend +12")
        elif trend == 'Downtrend':
            score -= 10; reasons.append("Downtrend −10")
        else:
            score += 2; reasons.append("Sideways +2")

        strength = str(stock.get('trend_strength', 'Neutral'))
        if strength.lower() == 'strong':
            score += 6; reasons.append("Strong trend +6")
        elif strength.lower() == 'weak':
            score -= 2; reasons.append("Weak trend −2")

        # RSI signals
        rsi_signal = _safe_get(stock, ['indicators', 'RSI', 'signal'], 'Neutral')
        if rsi_signal == 'Oversold':
            score += 6; reasons.append("RSI Oversold +6")
        elif rsi_signal == 'Overbought':
            score -= 6; reasons.append("RSI Overbought −6")

        # MACD trend
        macd_trend = _safe_get(stock, ['indicators', 'MACD', 'Trend'], 'Neutral')
        if macd_trend == 'Bullish':
            score += 9; reasons.append("MACD Bullish +9")
        elif macd_trend == 'Bearish':
            score -= 9; reasons.append("MACD Bearish −9")

        # SMA 20/50 cross trend
        sma_trend = _safe_get(stock, ['indicators', 'SMA', 'Trend'], 'Neutral')
        if sma_trend == 'Bullish':
            score += 6; reasons.append("SMA Bullish +6")
        elif sma_trend == 'Bearish':
            score -= 6; reasons.append("SMA Bearish −6")

        # Volume context
        vol_trend = _safe_get(stock, ['indicators', 'Volume', 'Trend'], 'Neutral')
        if isinstance(vol_trend, str) and vol_trend.lower().startswith('above'):
            score += 4; reasons.append("Volume above avg +4")

        # Volatility penalty for very high vol
        volatility = float(_safe_get(stock, ['indicators'], {}).get('Volatility', 0) or 0)
        if volatility >= 5.0:
            score -= 6; reasons.append("Volatility ≥5% −6")
        elif volatility >= 3.0:
            score -= 3; reasons.append("Volatility ≥3% −3")

        # Action preference bias
        action = str(stock.get('action', 'Watch'))
        if prefer_action and action == prefer_action:
            score += 4; reasons.append(f"Prefers {prefer_action} +4")

        # Clamp and annotate
        score = max(0.0, min(100.0, score))
        stock = dict(stock)  # do not mutate input list entries
        stock['conviction'] = {
            'score': round(score, 1),
            'pass': score >= float(min_score),
            'reasons': reasons,
            'threshold': float(min_score),
        }
        scored.append(stock)

    # Filter and sort by score desc
    passed = [s for s in scored if s['conviction']['pass']]
    passed.sort(key=lambda s: s['conviction']['score'], reverse=True)
    return passed


def horizon_recommendation(stock: Dict[str, Any]) -> Dict[str, Any]:
    """Compute short/medium/long-term recommendations with confidence and alignment.

    Heuristic weights using available indicators; robust to missing data.
    Returns a dict with keys: alignment, short_term, medium_term, long_term, overall.
    """
    # Helpers
    def get(path, default=None):
        return _safe_get(stock, path, default)

    # Extract common signals
    trend = str(stock.get('trend', 'Sideways'))
    strength = str(stock.get('trend_strength', 'Neutral')).lower()
    rsi_signal = get(['indicators', 'RSI', 'signal'], 'Neutral')
    macd_trend = get(['indicators', 'MACD', 'Trend'], 'Neutral')
    sma_trend = get(['indicators', 'SMA', 'Trend'], 'Neutral')
    rr = float(stock.get('risk_reward', 0) or 0)
    vol = float(get(['indicators'], {}).get('Volatility', 0) or 0)
    health = float(stock.get('health_score', 0) or 0)

    def decide_short():
        buy, sell = 0.0, 0.0
        if rsi_signal == 'Oversold': buy += 15
        if rsi_signal == 'Overbought': sell += 15
        if macd_trend == 'Bullish': buy += 12
        if macd_trend == 'Bearish': sell += 12
        if rr >= 2.0: buy += 6
        if rr < 1.0: sell += 6
        if vol >= 5.0: buy -= 4; sell -= 4  # uncertainty penalty
        if trend == 'Uptrend': buy += 4
        if trend == 'Downtrend': sell += 4
        if strength == 'strong': buy += 2 if buy >= sell else 0; sell += 2 if sell > buy else 0
        if abs(buy - sell) < 8: action = 'Hold'
        else: action = 'Buy' if buy > sell else 'Sell'
        conf = max(10.0, min(95.0, max(buy, sell)))
        return {'action': action, 'confidence': round(conf, 1)}

    def decide_medium():
        buy, sell = 0.0, 0.0
        if sma_trend == 'Bullish': buy += 14
        if sma_trend == 'Bearish': sell += 14
        if trend == 'Uptrend': buy += 10
        if trend == 'Downtrend': sell += 10
        if rr >= 1.5: buy += 6
        if rr < 1.0: sell += 6
        if health >= 60: buy += 5
        if health < 40: sell += 5
        if macd_trend == 'Bullish': buy += 4
        if macd_trend == 'Bearish': sell += 4
        if abs(buy - sell) < 8: action = 'Hold'
        else: action = 'Buy' if buy > sell else 'Sell'
        conf = max(10.0, min(95.0, max(buy, sell)))
        return {'action': action, 'confidence': round(conf, 1)}

    def decide_long():
        buy, sell = 0.0, 0.0
        if trend == 'Uptrend': buy += 12
        if trend == 'Downtrend': sell += 12
        if health >= 70: buy += 15
        if health < 40: sell += 15
        if sma_trend == 'Bullish': buy += 8
        if sma_trend == 'Bearish': sell += 8
        # Favor balanced risk/reward for long-term
        if rr >= 1.5: buy += 5
        if rr < 1.0: sell += 5
        if abs(buy - sell) < 8: action = 'Hold'
        else: action = 'Buy' if buy > sell else 'Sell'
        conf = max(10.0, min(95.0, max(buy, sell)))
        return {'action': action, 'confidence': round(conf, 1)}

    short = decide_short()
    medium = decide_medium()
    long = decide_long()

    actions = [short['action'], medium['action'], long['action']]
    same = len({a for a in actions if a != 'Hold'})
    if actions.count('Buy') >= 2 or actions.count('Sell') >= 2:
        alignment = 100 if len(set(actions)) == 1 else 75
    elif same >= 2:
        alignment = 66
    else:
        alignment = 33

    overall_conf = round((short['confidence'] + medium['confidence'] + long['confidence']) / 3.0, 1)
    overall_action = max(['Buy', 'Hold', 'Sell'], key=lambda a: actions.count(a))

    return {
        'alignment': alignment,
        'short_term': short,
        'medium_term': medium,
        'long_term': long,
        'overall': {'action': overall_action, 'confidence': overall_conf}
    }


def annotate_horizon_advice(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotate each stock with horizon-based recommendations under `advice`."""
    annotated = []
    for s in stocks:
        st = dict(s)
        try:
            st['advice'] = horizon_recommendation(st)
        except Exception:
            st['advice'] = {
                'alignment': 0,
                'short_term': {'action': 'Hold', 'confidence': 0},
                'medium_term': {'action': 'Hold', 'confidence': 0},
                'long_term': {'action': 'Hold', 'confidence': 0},
                'overall': {'action': 'Hold', 'confidence': 0}
            }
        annotated.append(st)
    return annotated


# ----------------- Plutus Analyzer (Ranking Picks) -----------------

def _score_for_horizon(stock: Dict[str, Any], horizon: str) -> float:
    """Composite score for a given horizon using different emphases per horizon.

    - short_term: momentum (5d), MACD histogram slope, RSI, volume trend, RR
    - medium_term: SMA trend, 1M momentum, MACD trend, RR, volatility
    - long_term: trend + strength, 3M momentum, health score, RR, volatility
    """
    get = lambda *p, d=None: _safe_get(stock, list(p), d)
    conv = float(stock.get('conviction', {}).get('score', 0) or 0)
    hrec = stock.get('advice', {}).get(horizon, {}) if isinstance(stock.get('advice'), dict) else {}
    conf = float(hrec.get('confidence', 0) or 0)
    rr = float(stock.get('risk_reward', 0) or 0)
    vol = float(get('indicators', 'Volatility') or 0)
    rsi_sig = get('indicators', 'RSI', 'signal', d='Neutral')
    macd_tr = get('indicators', 'MACD', 'Trend', d='Neutral')
    sma_tr = get('indicators', 'SMA', 'Trend', d='Neutral')
    trend = str(stock.get('trend', 'Sideways'))
    strength = str(stock.get('trend_strength', 'Neutral')).lower()
    m = stock.get('metrics', {}) if isinstance(stock.get('metrics'), dict) else {}
    ret_5 = float(m.get('ret_5d', 0) or 0) * 100
    ret_21 = float(m.get('ret_21d', 0) or 0) * 100
    ret_63 = float(m.get('ret_63d', 0) or 0) * 100
    hist_slope = float(m.get('macd_hist_slope', 0) or 0)
    dist_r1 = float(m.get('dist_r1', 0) or 0)
    dist_s1 = float(m.get('dist_s1', 0) or 0)
    health = float(stock.get('health_score', 0) or 0)

    score = 0.0

    if horizon == 'short_term':
        score += 0.35 * conv + 0.35 * conf + 6.0 * (rr - 1.0)
        # Momentum + breakouts
        score += min(max(ret_5, -5), 5)  # clamp contribution
        if hist_slope > 0: score += 4
        if macd_tr == 'Bullish': score += 3
        if rsi_sig == 'Oversold': score += 2
        if rsi_sig == 'Overbought': score -= 3
        # Close to breakout (small positive distance to R1)
        if 0 < dist_r1 < 0.03: score += 3
        if dist_s1 < 0.02: score -= 2  # tight to stop
        # Volatility penalty
        if vol > 5: score -= 4
        elif vol > 3: score -= 2

    elif horizon == 'medium_term':
        score += 0.4 * conv + 0.35 * conf + 9.0 * (rr - 1.0)
        if sma_tr == 'Bullish': score += 5
        if trend == 'Uptrend': score += 3
        score += min(max(ret_21, -10), 10) * 0.4
        if macd_tr == 'Bullish': score += 2
        if vol > 5: score -= 5
        elif vol > 3: score -= 3

    else:  # long_term
        score += 0.45 * conv + 0.3 * conf + 7.0 * (rr - 1.0)
        if trend == 'Uptrend': score += 5
        if strength == 'strong': score += 3
        score += min(max(ret_63, -20), 20) * 0.3
        score += (health - 50) * 0.15  # center at 50
        if vol > 5: score -= 6
        elif vol > 3: score -= 4

    return round(score, 2)


def generate_picks(stocks: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    """Generate top Buy/Sell picks for short/medium/long horizons.

    Returns a dict:
        {
          'short': {'buy': [...], 'sell': [...]},
          'medium': {'buy': [...], 'sell': [...]},
          'long': {'buy': [...], 'sell': [...]}
        }
    Each item carries: symbol, price, score, confidence, action, rr, trend, rsi/macd summaries.
    """
    if not stocks:
        return {'short': {'buy': [], 'sell': []}, 'medium': {'buy': [], 'sell': []}, 'long': {'buy': [], 'sell': []}}

    # Ensure advice exists
    enriched = annotate_horizon_advice(stocks)

    def extract(stock, horizon_key):
        ind = stock.get('indicators', {})
        rsi = ind.get('RSI', {})
        macd = ind.get('MACD', {})
        return {
            'symbol': stock['symbol'],
            'name': stock.get('name') or (stock['symbol'].split('.')[0] if isinstance(stock.get('symbol'), str) else stock.get('symbol')),
            'price': float(stock.get('price', 0) or 0),
            'action': stock.get('advice', {}).get(horizon_key, {}).get('action', 'Hold'),
            'confidence': float(stock.get('advice', {}).get(horizon_key, {}).get('confidence', 0) or 0),
            'score': _score_for_horizon(stock, horizon_key),
            'rr': float(stock.get('risk_reward', 0) or 0),
            'trend': str(stock.get('trend', 'Sideways')),
            'rsi': {'value': float(rsi.get('value', 50) or 50), 'signal': rsi.get('signal', 'Neutral')},
            'macd': {'trend': macd.get('Trend', 'Neutral')},
            'reason': stock.get('action_reason', ''),
            'link': f"/analyze?symbols={stock['symbol']}"
        }

    def horizon_filters(item, hkey, stock):
        ind = stock.get('indicators', {})
        rsi_sig = ind.get('RSI', {}).get('signal')
        macd_tr = ind.get('MACD', {}).get('Trend')
        sma_tr = ind.get('SMA', {}).get('Trend')
        trend = stock.get('trend')
        rr = float(stock.get('risk_reward', 0) or 0)
        health = float(stock.get('health_score', 0) or 0)
        m = stock.get('metrics', {})
        hist_slope = float(m.get('macd_hist_slope', 0) or 0)
        if hkey == 'short_term':
            ok_buy = (item['action'] == 'Buy') and (hist_slope > 0 or rsi_sig != 'Overbought')
            ok_sell = (item['action'] == 'Sell')
        elif hkey == 'medium_term':
            ok_buy = (item['action'] == 'Buy') and ((sma_tr == 'Bullish') or (trend == 'Uptrend'))
            ok_sell = (item['action'] == 'Sell')
        else:
            ok_buy = (item['action'] == 'Buy') and (trend == 'Uptrend') and (health >= 60) and (rr >= 1.3)
            ok_sell = (item['action'] == 'Sell')
        return ok_buy, ok_sell

    def rank(hkey):
        items = []
        for s in enriched:
            it = extract(s, hkey)
            items.append((it, s))
        buy_items = []
        sell_items = []
        for it, s in items:
            ok_buy, ok_sell = horizon_filters(it, hkey, s)
            if ok_buy:
                buy_items.append(it)
            if ok_sell:
                sell_items.append(it)
        buys = sorted(buy_items, key=lambda x: (x['score'], x['confidence'], x['rr']), reverse=True)[:top_n]
        sells = sorted(sell_items, key=lambda x: (x['score'], x['confidence']), reverse=True)[:top_n]
        return {'buy': buys, 'sell': sells}

    return {
        'short': rank('short_term'),
        'medium': rank('medium_term'),
        'long': rank('long_term')
    }
