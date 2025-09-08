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
        for i, key in enumerate(path):
            if not isinstance(cur, dict):
                return default
            cur = cur.get(key, None)
            if cur is None and i < len(path) - 1:
                return default
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

    # Normalize preference once (case-insensitive)
    pref = (prefer_action or "").strip().lower() if isinstance(prefer_action, str) else None

    for stock in stocks or []:
        reasons: List[str] = []
        score = 0.0

        # 1) Health score (0–100) up to 30 pts (smooth)
        try:
            health = float(stock.get('health_score', 0) or 0.0)
        except Exception:
            health = 0.0
        health_clamped = max(0.0, min(100.0, health))
        health_pts = 0.30 * health_clamped
        score += health_pts
        reasons.append(f"Health {int(health_clamped)} contributes {health_pts:.0f}")

        # 2) Risk/Reward weighting (diminishing beyond 2.5; lighter penalty <1.0 if strong trend)
        try:
            rr = float(stock.get('risk_reward', 0) or 0.0)
        except Exception:
            rr = 0.0

        # Determine trend context for adaptive penalty
        trend_ctx = str(stock.get('trend', 'Sideways'))
        macd_tr = _safe_get(stock, ['indicators', 'MACD', 'Trend'], 'Neutral')

        if rr >= 3.0:
            score += 19; reasons.append("Risk/Reward ≥3.0 +19")
        elif rr >= 2.5:
            score += 18; reasons.append("Risk/Reward ≥2.5 +18")
        elif rr >= 2.0:
            score += 14; reasons.append("Risk/Reward ≥2.0 +14")
        elif rr >= 1.5:
            score += 8; reasons.append("Risk/Reward ≥1.5 +8")
        elif rr >= 1.0:
            score += 3; reasons.append("Risk/Reward ≥1.0 +3")
        else:
            # If strong directional context, soften the penalty a touch
            softener = 4 if (trend_ctx == 'Uptrend' and macd_tr == 'Bullish') or (trend_ctx == 'Downtrend' and macd_tr == 'Bearish') else 0
            score -= max(12 - softener, 6)
            reasons.append(f"Risk/Reward <1.0 −{max(12 - softener, 6)}")

        # 3) Trend direction and strength
        trend = str(trend_ctx)
        if trend == 'Uptrend':
            score += 12; reasons.append("Uptrend +12")
        elif trend == 'Downtrend':
            score -= 10; reasons.append("Downtrend −10")
        else:
            score += 2; reasons.append("Sideways +2")

        strength = str(stock.get('trend_strength', 'Neutral'))
        s = strength.strip().lower()
        if s == 'strong':
            score += 6; reasons.append("Strong trend +6")
        elif s == 'weak':
            score -= 2; reasons.append("Weak trend −2")

        # 4) RSI signals
        rsi_signal = _safe_get(stock, ['indicators', 'RSI', 'signal'], 'Neutral')
        if rsi_signal == 'Oversold':
            score += 6; reasons.append("RSI Oversold +6")
        elif rsi_signal == 'Overbought':
            score -= 6; reasons.append("RSI Overbought −6")

        # 5) MACD trend
        if macd_tr == 'Bullish':
            score += 9; reasons.append("MACD Bullish +9")
        elif macd_tr == 'Bearish':
            score -= 9; reasons.append("MACD Bearish −9")

        # 6) SMA 20/50 cross trend
        sma_trend = _safe_get(stock, ['indicators', 'SMA', 'Trend'], 'Neutral')
        if sma_trend == 'Bullish':
            score += 6; reasons.append("SMA Bullish +6")
        elif sma_trend == 'Bearish':
            score -= 6; reasons.append("SMA Bearish −6")

        # 7) Volume context (reward only; avoid penalizing low volume twice)
        vol_trend = _safe_get(stock, ['indicators', 'Volume', 'Trend'], 'Neutral')
        if isinstance(vol_trend, str) and vol_trend.strip().lower().startswith('above'):
            score += 4; reasons.append("Volume above avg +4")

        # 8) Volatility penalty bands (smooth)
        try:
            volatility = float((_safe_get(stock, ['indicators'], {}) or {}).get('Volatility', 0) or 0.0)
        except Exception:
            volatility = 0.0
        if volatility >= 5.0:
            score -= 6; reasons.append("Volatility ≥5% −6")
        elif volatility >= 3.0:
            score -= 3; reasons.append("Volatility ≥3% −3")

        # 9) Action preference bias (case-insensitive)
        action = str(stock.get('action', 'Watch'))
        if pref and action.strip().lower() == pref:
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
        # uncertainty penalty (don’t bias direction, reduce both)
        if vol >= 5.0: buy -= 4; sell -= 4
        if trend == 'Uptrend': buy += 4
        if trend == 'Downtrend': sell += 4
        if strength == 'strong':
            if buy >= sell: buy += 2
            else: sell += 2
        action = 'Hold' if abs(buy - sell) < 8 else ('Buy' if buy > sell else 'Sell')
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
        action = 'Hold' if abs(buy - sell) < 8 else ('Buy' if buy > sell else 'Sell')
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
        if rr >= 1.5: buy += 5
        if rr < 1.0: sell += 5
        action = 'Hold' if abs(buy - sell) < 8 else ('Buy' if buy > sell else 'Sell')
        conf = max(10.0, min(95.0, max(buy, sell)))
        return {'action': action, 'confidence': round(conf, 1)}

    short = decide_short()
    medium = decide_medium()
    long = decide_long()

    actions = [short['action'], medium['action'], long['action']]
    # Prefer 'Hold' on ties between Buy/Sell; otherwise use majority
    buy_count = actions.count('Buy')
    sell_count = actions.count('Sell')
    if buy_count == sell_count and buy_count >= 1:
        overall_action = 'Hold'
    else:
        overall_action = 'Buy' if buy_count > sell_count else ('Sell' if sell_count > buy_count else 'Hold')

    # Alignment: 100 if all equal and not Hold; 75 if 2 agree; 66 if 2 non-Hold agree but 3rd Hold; else 33
    if len(set(actions)) == 1 and actions[0] != 'Hold':
        alignment = 100
    elif buy_count >= 2 or sell_count >= 2:
        alignment = 75
    elif len({a for a in actions if a != 'Hold'}) >= 1 and (buy_count == 2 or sell_count == 2 or (buy_count == 1 and sell_count == 1)):
        alignment = 66
    else:
        alignment = 33

    overall_conf = round((short['confidence'] + medium['confidence'] + long['confidence']) / 3.0, 1)

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
        # Momentum + breakouts (clamped)
        score += max(-5.0, min(5.0, ret_5))
        if hist_slope > 0: score += 4
        if macd_tr == 'Bullish': score += 3
        if rsi_sig == 'Oversold': score += 2
        if rsi_sig == 'Overbought': score -= 3
        # Close to breakout (but avoid if too close to support)
        if 0 < dist_r1 < 0.03: score += 3
        if dist_s1 < 0.02: score -= 2
        # Volatility penalty
        if vol > 5: score -= 4
        elif vol > 3: score -= 2

    elif horizon == 'medium_term':
        score += 0.4 * conv + 0.35 * conf + 9.0 * (rr - 1.0)
        if sma_tr == 'Bullish': score += 5
        if trend == 'Uptrend': score += 3
        score += 0.4 * max(-10.0, min(10.0, ret_21))
        if macd_tr == 'Bullish': score += 2
        if vol > 5: score -= 5
        elif vol > 3: score -= 3

    else:  # long_term
        score += 0.45 * conv + 0.3 * conf + 7.0 * (rr - 1.0)
        if trend == 'Uptrend': score += 5
        if strength == 'strong': score += 3
        score += 0.3 * max(-20.0, min(20.0, ret_63))
        score += 0.15 * (health - 50.0)
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
        # Sparkline from chart candles (close-only, last 30)
        spark = []
        try:
            candles = (stock.get('chart') or {}).get('candles') or []
            if candles:
                closes = []
                for c in candles[-30:]:
                    try:
                        closes.append(float(c['y'][3]))
                    except Exception:
                        continue
                spark = closes
        except Exception:
            spark = []
        # Performance chips from metrics (1M/3M)
        m = stock.get('metrics', {}) if isinstance(stock.get('metrics'), dict) else {}
        try:
            ret1m = float(m.get('ret_21d', 0) or 0) * 100.0
        except Exception:
            ret1m = 0.0
        try:
            ret3m = float(m.get('ret_63d', 0) or 0) * 100.0
        except Exception:
            ret3m = 0.0
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
            'link': f"/analyze?symbols={stock['symbol']}",
            'spark': spark,
            'ret1m': ret1m,
            'ret3m': ret3m,
            'horizon': horizon_key
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


# ----------------- LLM Enrichment (Ollama / qwen2:7b) -----------------

def enrich_picks_with_llm(picks: Dict[str, Any], universe: str | None = None) -> Dict[str, Any]:
    """
    Add concise natural-language rationales to each pick using a local Ollama model.
    - Honors env OLLAMA_MODEL; defaults to 'qwen2:7b'
    - Talks to http://localhost:11434 (Ollama default)
    - Non-fatal: on any exception, returns original picks unchanged

    The structure of `picks` is preserved; each item gets 'llm_reason' text.
    """
    import os, json
    from textwrap import dedent

    model = os.getenv("OLLAMA_MODEL", "qwen2:7b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "18"))  # seconds
    endpoint = f"{host.rstrip('/')}/api/generate"

    try:
        import requests
    except Exception:
        # requests not available; return unchanged
        return picks

    def _brief(item: Dict[str, Any]) -> str:
        # Keep it deterministic and compact for UI chips
        return dedent(f"""
        Symbol: {item.get('symbol')}
        Name: {item.get('name')}
        Horizon: {item.get('horizon')}
        Action: {item.get('action')}
        Confidence: {item.get('confidence')}
        Score: {item.get('score')}
        Price: {item.get('price')}
        RR: {item.get('rr')}
        Trend: {item.get('trend')}
        RSI: {((item.get('rsi') or {}).get('value'))} ({((item.get('rsi') or {}).get('signal'))})
        MACD Trend: {((item.get('macd') or {}).get('trend'))}
        1M Ret%: {item.get('ret1m')}
        3M Ret%: {item.get('ret3m')}
        """).strip()

    def _prompt(items: List[Dict[str, Any]], horizon: str) -> str:
        bullets = "\n\n".join(_brief(x) for x in items)
        return dedent(f"""
        You are an equity trading assistant. Write ONE short rationale (max 2 sentences, ~40 words)
        per item explaining why the suggested action is reasonable, citing 2–3 concrete signals
        (trend/RSI/MACD/RR/momentum). Avoid disclaimers and generic language. No emojis.

        Universe: {universe or 'unspecified'}
        Horizon: {horizon}

        Items:
        {bullets}

        Respond as a JSON list of strings, in the SAME order as items, length = {len(items)}.
        """).strip()

    def _call_llm(items: List[Dict[str, Any]], horizon: str) -> List[str]:
        if not items:
            return []
        payload = {
            "model": model,
            "prompt": _prompt(items, horizon),
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            # Expect JSON list; be forgiving
            try:
                arr = json.loads(text)
                if isinstance(arr, list) and len(arr) == len(items):
                    return [str(x) for x in arr]
            except Exception:
                pass
            # Fallback: split by newline
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if len(lines) >= len(items):
                return lines[:len(items)]
            # Otherwise pad
            return (lines + [""] * len(items))[:len(items)]
        except Exception:
            return [""] * len(items)

    # Enrich each bucket
    out = {k: {"buy": list(v.get("buy", [])), "sell": list(v.get("sell", []))}
           for k, v in picks.items()}

    for horizon_key in ("short", "medium", "long"):
        for side in ("buy", "sell"):
            items = out.get(horizon_key, {}).get(side, [])
            reasons = _call_llm(items, horizon_key)
            for i, r in enumerate(reasons):
                try:
                    items[i]["llm_reason"] = r.strip()
                except Exception:
                    pass

    return out
