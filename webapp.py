"""
Flask web application for stock portfolio analysis
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from data_fetcher import get_nifty50_data
from stock_analyzer import format_stock_analysis
from config import NIFTY50_SYMBOLS, NIFTY500_SYMBOLS, POSITIONS
import os
from ai_analysis import conviction_gate, annotate_horizon_advice, generate_picks, enrich_picks_with_llm
from reversals import rank_reversals
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-me')
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Enable template auto-reload
app.state = {
    'watchlist': set(),
    'alerts': []
}

@app.after_request
def after_request(response):
    """Ensure proper headers are set for all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def get_market_insights(symbols):
    """Get overall market insights for a given universe of symbols.

    To keep the homepage snappy, we sample a small subset of the universe.
    """
    # Sample a subset for speed (12 is a sweet spot for UI responsiveness)
    # Use a deterministic daily sample so numbers shift day-to-day and differ by universe
    try:
        import random
        syms = list(symbols)
        k = 12 if len(syms) > 12 else len(syms)
        seed = int(datetime.utcnow().strftime('%Y%m%d'))
        r = random.Random(seed)
        subset = r.sample(syms, k) if k and len(syms) >= k else syms
    except Exception:
        subset = symbols[:12] if isinstance(symbols, list) else symbols
    market_data = get_nifty50_data(subset)
    analysis = format_stock_analysis(market_data)

    avg_health = (sum(stock.get('health_score', 0) for stock in analysis) / max(1, len(analysis))) if analysis else 0
    insights = {
        'sentiment': 'Positive' if avg_health > 50 else 'Negative',
        'top_movers': len([stock for stock in analysis if abs(stock.get('day_change', 0)) > 2]),
        'buy_signals': len([stock for stock in analysis if stock.get('action') == 'Buy']),
        'sell_signals': len([stock for stock in analysis if stock.get('action') == 'Sell'])
    }
    return insights, analysis

def compute_kpis(analysis, positions):
    """Compute portfolio KPIs (value, today's % change) from holdings and analysis.

    Today % = sum(qty * price * day_change%) / sum(qty * price)
    """
    price_map = {s['symbol']: s for s in analysis}
    total_value = 0.0
    weighted_change_value = 0.0
    for p in positions:
        sym = p.get('symbol')
        qty = float(p.get('qty', 0) or 0)
        info = price_map.get(sym)
        if not info or qty <= 0:
            continue
        price = float(info.get('price', 0) or 0)
        day_pct = float(info.get('day_change', 0) or 0)  # percent
        position_value = qty * price
        total_value += position_value
        weighted_change_value += position_value * (day_pct / 100.0)
    today_pct = (weighted_change_value / total_value * 100.0) if total_value > 0 else 0.0
    return {'value': round(total_value, 2), 'today_pct': round(today_pct, 2)}

@app.route('/')
def index():
    """Render the main page with market insights and featured analysis."""
    # Universe selector: u=n50|n500|watch
    u = (request.args.get('u') or 'n50').lower()
    if u == 'n500':
        universe_symbols = NIFTY500_SYMBOLS
    elif u == 'watch':
        watch = sorted(list(app.state['watchlist']))
        universe_symbols = watch if watch else NIFTY50_SYMBOLS
    else:
        universe_symbols = NIFTY50_SYMBOLS

    insights, analysis = get_market_insights(universe_symbols)
    try:
        from ai_analysis import conviction_gate
        featured = conviction_gate(analysis, min_score=0)[:6]
    except Exception:
        featured = analysis[:6]

    # Compute KPIs from configured positions
    try:
        pos_syms = [p['symbol'] for p in POSITIONS]
        pos_results = get_nifty50_data(pos_syms)
        pos_analysis = format_stock_analysis(pos_results)
        kpi = compute_kpis(pos_analysis, POSITIONS)
    except Exception:
        kpi = {'value': 0.0, 'today_pct': 0.0}

    return render_template('index.html', insights=insights, stocks=featured, kpi=kpi, universe=u)

@app.route('/analyze/universe')
def analyze_universe():
    """Analyze a full universe (NIFTY50/NIFTY500/Watchlist) and render a grid of stock cards.

    Query params:
      - u: n50|n500|watch (default n50)
      - limit: int (default 24)
      - sample: 0/1 (if 1, randomly sample the universe before limiting)
      - min_score: optional conviction threshold to display
    """
    u = (request.args.get('u') or 'n50').lower()
    try:
        limit = int(request.args.get('limit', 24))
    except Exception:
        limit = 24
    sample = request.args.get('sample', '0') in ('1', 'true', 'yes')
    try:
        min_score = float(request.args.get('min_score', 0) or 0)
    except Exception:
        min_score = 0.0

    if u == 'n500':
        symbols = NIFTY500_SYMBOLS
    elif u == 'watch':
        wl = sorted(list(app.state['watchlist']))
        symbols = wl if wl else NIFTY50_SYMBOLS
    else:
        symbols = NIFTY50_SYMBOLS

    # Build symbol set respecting sampling/limit
    syms = list(symbols)
    if sample and len(syms) > limit:
        import random
        syms = random.sample(syms, limit)
    else:
        syms = syms[:limit]

    # Run analysis and score
    raw = get_nifty50_data(syms)
    formatted = format_stock_analysis(raw)
    scored = conviction_gate(formatted, min_score=0)
    # Apply threshold if provided
    if min_score > 0:
        scored = [s for s in scored if s.get('conviction', {}).get('score', 0) >= min_score]
    # Sort by conviction desc
    scored.sort(key=lambda s: s.get('conviction', {}).get('score', 0), reverse=True)

    # KPIs for header
    try:
        pos_results = get_nifty50_data([p['symbol'] for p in POSITIONS])
        kpi = compute_kpis(format_stock_analysis(pos_results), POSITIONS)
    except Exception:
        kpi = {'value': 0.0, 'today_pct': 0.0}

    # Render using analysis grid mode
    return render_template('analysis.html', stocks=scored, min_score=min_score, now=datetime.now(), kpi=kpi, grid_mode=True)

@app.route('/movers')
def movers_view():
    """Top Movers dashboard (biggest +/- day change) with quick actions.

    Query params:
      - u: n50|n500|watch (default n50)
      - limit: int (default 50 per side)
      - sample: 0/1 (sample the universe before ranking)
    """
    u = (request.args.get('u') or 'n50').lower()
    try:
        per_side = int(request.args.get('limit', 50))
    except Exception:
        per_side = 50
    sample = request.args.get('sample', '0') in ('1', 'true', 'yes')

    if u == 'n500':
        symbols = NIFTY500_SYMBOLS
    elif u == 'watch':
        wl = sorted(list(app.state['watchlist']))
        symbols = wl if wl else NIFTY50_SYMBOLS
    else:
        symbols = NIFTY50_SYMBOLS

    syms = list(symbols)
    # Optional sampling to keep fetching fast on large universes
    if sample and len(syms) > per_side * 4:
        import random
        syms = random.sample(syms, per_side * 4)

    raw = get_nifty50_data(syms)
    analysis = format_stock_analysis(raw)

    # Rank by absolute day change
    ranked = sorted(
        analysis,
        key=lambda s: abs(float(s.get('day_change', 0) or 0)),
        reverse=True
    )
    gainers = [s for s in ranked if (s.get('day_change', 0) or 0) > 0][:per_side]
    losers = [s for s in ranked if (s.get('day_change', 0) or 0) < 0][:per_side]

    # KPIs for header
    try:
        pos_results = get_nifty50_data([p['symbol'] for p in POSITIONS])
        kpi = compute_kpis(format_stock_analysis(pos_results), POSITIONS)
    except Exception:
        kpi = {'value': 0.0, 'today_pct': 0.0}

    return render_template('movers.html', gainers=gainers, losers=losers, kpi=kpi, universe=u)

@app.route('/api/movers')
def movers_api():
    u = (request.args.get('u') or 'n50').lower()
    try:
        limit = int(request.args.get('limit', 50))
    except Exception:
        limit = 50
    if u == 'n500':
        symbols = NIFTY500_SYMBOLS
    elif u == 'watch':
        wl = sorted(list(app.state['watchlist']))
        symbols = wl if wl else NIFTY50_SYMBOLS
    else:
        symbols = NIFTY50_SYMBOLS
    raw = get_nifty50_data(symbols)
    analysis = format_stock_analysis(raw)
    ranked = sorted(analysis, key=lambda s: abs(float(s.get('day_change', 0) or 0)), reverse=True)
    gainers = [s for s in ranked if (s.get('day_change', 0) or 0) > 0][:limit]
    losers = [s for s in ranked if (s.get('day_change', 0) or 0) < 0][:limit]
    return jsonify({'gainers': gainers, 'losers': losers})


# ---------------------- AI Pipeline (conviction, horizons, picks) ----------------------

def _resolve_universe(param: str):
    """
    Return (universe_key, symbols) without any fallback.
    - 'n50'  -> NIFTY50_SYMBOLS
    - 'n500' -> NIFTY500_SYMBOLS
    - 'watch'-> exact watchlist (may be empty)
    Any other value -> ('n50', NIFTY50_SYMBOLS) as a *default only
    when param missing/invalid* (not a fallback for watch).
    """
    u = (param or 'n50').lower()
    if u == 'n500':
        return 'n500', NIFTY500_SYMBOLS
    if u == 'watch':
        return 'watch', sorted(list(app.state['watchlist']))
    return 'n50', NIFTY50_SYMBOLS



@app.route('/api/ai/pipeline', methods=['GET', 'POST'])
def api_ai_pipeline():
    """Run the full AI analysis pipeline and return structured JSON.

    GET params (optional):
      - u: n50|n500|watch (default n50)
      - limit: cap number of symbols (default 24)
      - sample: 0/1 sample universe before capping (default 0)
      - llm: 0/1 include Ollama rationales if model configured (default off unless env set)

    POST body (optional): {"symbols": [..], "llm": 0/1}
      - If symbols provided, universe params are ignored.
    """
    symbols = None
    use_llm = False
    if request.method == 'POST' and request.is_json:
        data = request.get_json(silent=True) or {}
        symbols = [s.strip().upper() for s in (data.get('symbols') or []) if isinstance(s, str) and s.strip()]
        use_llm = bool(os.getenv('OLLAMA_MODEL')) or (str(data.get('llm', '0')) in ('1', 'true', 'yes'))
    if not symbols:
        u, universe = _resolve_universe(request.args.get('u'))
        try:
            limit = int(request.args.get('limit', 24))
        except Exception:
            limit = 24
        sample = request.args.get('sample', '0') in ('1', 'true', 'yes')
        syms = list(universe)
        if sample and len(syms) > limit:
            import random
            symbols = random.sample(syms, limit)
        else:
            symbols = syms[:limit]
        use_llm = bool(os.getenv('OLLAMA_MODEL')) or (request.args.get('llm') == '1')
    # Fetch and analyze
    raw = get_nifty50_data(symbols)
    formatted = format_stock_analysis(raw)
    scored = conviction_gate(formatted, min_score=0)
    with_advice = annotate_horizon_advice(scored)
    picks = generate_picks(with_advice, top_n=5)
    if use_llm:
        try:
            picks = enrich_picks_with_llm(picks)
        except Exception:
            pass
    return jsonify({
        'count': len(with_advice),
        'symbols': [s.get('symbol') for s in with_advice],
        'stocks': with_advice,
        'picks': picks,
        'llm': use_llm
    })

@app.route('/picks')
def picks_view():
    """Render Plutus Picks page with Buy/Sell lists for S/M/L terms."""
    # Universe selector: u=n50|n500|watch
    u = (request.args.get('u') or 'n50').lower()
    if u == 'n500':
        symbols = NIFTY500_SYMBOLS
    elif u == 'watch':
        wl = sorted(list(app.state['watchlist']))
        symbols = wl if wl else NIFTY50_SYMBOLS
    else:
        symbols = NIFTY50_SYMBOLS

    market_data = get_nifty50_data(symbols)
    analysis = format_stock_analysis(market_data)
    scored = conviction_gate(analysis, min_score=0)
    scored = annotate_horizon_advice(scored)
    picks = generate_picks(scored, top_n=5)
    # Optional: LLM rationales (enable via OLLAMA_MODEL or ?llm=1)
    use_llm = bool(os.getenv('OLLAMA_MODEL')) or (request.args.get('llm') == '1')
    if use_llm:
        try:
            picks = enrich_picks_with_llm(picks, universe=u)
        except Exception:
            pass

    # KPIs for header
    try:
        pos_results = get_nifty50_data([p['symbol'] for p in POSITIONS])
        kpi = compute_kpis(format_stock_analysis(pos_results), POSITIONS)
    except Exception:
        kpi = {'value': 0.0, 'today_pct': 0.0}

    return render_template('picks.html', picks=picks, kpi=kpi, universe=u, llm_enabled=use_llm)

@app.route('/api/picks')
def picks_api():
    """JSON API for Plutus Picks."""
    u = (request.args.get('u') or 'n50').lower()
    if u == 'n500':
        symbols = NIFTY500_SYMBOLS
    elif u == 'watch':
        wl = sorted(list(app.state['watchlist']))
        symbols = wl if wl else NIFTY50_SYMBOLS


    market_data = get_nifty50_data(symbols)
    analysis = format_stock_analysis(market_data)
    scored = conviction_gate(analysis, min_score=0)
    scored = annotate_horizon_advice(scored)
    picks = generate_picks(scored, top_n=5)
    return jsonify(picks)

@app.route('/reversals')
def reversals_view():
    """Reversal dashboard: bullish/bearish candidates with reasons."""
    # Universe selector: u=n50|n500|watch
    u = (request.args.get('u') or 'n500').lower()
    if u == 'n50':
        symbols = NIFTY50_SYMBOLS
    elif u == 'watch':
        wl = sorted(list(app.state['watchlist']))
        symbols = wl if wl else NIFTY50_SYMBOLS
    else:
        symbols = NIFTY500_SYMBOLS

    raw = get_nifty50_data(symbols)
    formatted = format_stock_analysis(raw)
    rev = rank_reversals(raw, formatted, top_n=20)
    # KPIs for header
    try:
        pos_results = get_nifty50_data([p['symbol'] for p in POSITIONS])
        kpi = compute_kpis(format_stock_analysis(pos_results), POSITIONS)
    except Exception:
        kpi = {'value': 0.0, 'today_pct': 0.0}
    return render_template('reversals.html', reversals=rev, kpi=kpi, universe=u)

@app.route('/api/reversals')
def reversals_api():
    u = (request.args.get('u') or 'n500').lower()
    if u == 'n50':
        symbols = NIFTY50_SYMBOLS
    elif u == 'watch':
        wl = sorted(list(app.state['watchlist']))
        symbols = wl if wl else NIFTY50_SYMBOLS
    else:
        symbols = NIFTY500_SYMBOLS

    raw = get_nifty50_data(symbols)
    formatted = format_stock_analysis(raw)
    rev = rank_reversals(raw, formatted, top_n=50)
    return jsonify(rev)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Analyze stock(s) from search"""
    # Accept symbols via GET, form or JSON
    symbols: list[str] = []
    if request.method == 'GET':
        qs = request.args.get('symbols') or request.args.get('symbol') or ''
        symbols = [s.strip().upper() for s in qs.split(',') if s and s.strip()]
    elif request.is_json:
        payload = request.get_json(silent=True) or {}
        raw = payload.get('symbols') or payload.get('symbol') or ''
        if isinstance(raw, str):
            symbols = [raw]
        else:
            symbols = raw or []
        symbols = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    else:
        raw = request.form.get('symbols', '')
        symbols = [s.strip().upper() for s in raw.split(',') if s and s.strip()]
    
    if not symbols:
        # Non-JSON: redirect back home; JSON: return error payload
        if request.is_json:
            return jsonify({'error': 'Please enter at least one stock symbol'}), 400
        return redirect(url_for('index'))
    
    # Get analysis results
    results = get_nifty50_data(symbols)
    analysis = format_stock_analysis(results)

    # Compute conviction scores and sort (annotates each stock)
    scored = conviction_gate(analysis, min_score=0)
    scored = annotate_horizon_advice(scored)

    # Optional threshold filter
    try:
        threshold = float(request.args.get('min_score', request.form.get('min_score', 0)) or 0)
    except (TypeError, ValueError):
        threshold = 0
    filtered = [s for s in scored if s.get('conviction', {}).get('score', 0) >= threshold]

    # Compute KPIs for top bar
    try:
        pos_syms = [p['symbol'] for p in POSITIONS]
        pos_results = get_nifty50_data(pos_syms)
        pos_analysis = format_stock_analysis(pos_results)
        kpi = compute_kpis(pos_analysis, POSITIONS)
    except Exception:
        kpi = {'value': 0.0, 'today_pct': 0.0}

    # If JSON requested explicitly, return the first stock (compat for simple API)
    if request.is_json:
        first = (filtered[0] if filtered else (scored[0] if scored else {}))
        return jsonify(first)

    return render_template('analysis.html', stocks=filtered, min_score=threshold, now=datetime.now(), kpi=kpi)

@app.route('/api/watchlist', methods=['GET', 'POST'])
def api_watchlist():
    if request.method == 'GET':
        return jsonify(sorted(list(app.state['watchlist'])))
    data = request.get_json(silent=True) or {}
    symbol = str(data.get('symbol', '')).upper()
    action = data.get('action', 'add')
    if not symbol:
        return jsonify({'ok': False, 'error': 'symbol required'}), 400
    if action == 'remove':
        app.state['watchlist'].discard(symbol)
    else:
        app.state['watchlist'].add(symbol)
    return jsonify({'ok': True, 'watchlist': sorted(list(app.state['watchlist']))})

@app.route('/api/alerts', methods=['GET', 'POST'])
def api_alerts():
    if request.method == 'GET':
        return jsonify(app.state['alerts'])
    data = request.get_json(silent=True) or {}
    alert = {
        'symbol': str(data.get('symbol', '')).upper(),
        'condition': data.get('condition', 'above'),
        'price': float(data.get('price', 0) or 0)
    }
    if not alert['symbol'] or alert['price'] <= 0:
        return jsonify({'ok': False, 'error': 'invalid alert'}), 400
    app.state['alerts'].append(alert)
    return jsonify({'ok': True, 'alert': alert})

@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    """API endpoint for stock analysis"""
    if not request.is_json:
        return jsonify({'error': 'JSON body required'}), 400
    payload = request.get_json(silent=True) or {}
    raw = payload.get('symbols') or payload.get('symbol')
    if isinstance(raw, str):
        selected_symbols = [raw]
    else:
        selected_symbols = list(raw or [])
    selected_symbols = [str(s).strip().upper() for s in selected_symbols if isinstance(s, str) and s.strip()]
    if not selected_symbols:
        return jsonify({'error': 'Please select at least one stock'}), 400

    # Get analysis results
    results = get_nifty50_data(selected_symbols)
    analysis = format_stock_analysis(results)
    
    return jsonify({'stocks': analysis})

@app.route('/healthz')
def healthz():
    """Liveness probe endpoint."""
    try:
        return jsonify({'ok': True, 'time': datetime.utcnow().isoformat() + 'Z'}), 200
    except Exception:
        return jsonify({'ok': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
