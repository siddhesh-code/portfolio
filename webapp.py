"""
Flask web application for stock portfolio analysis
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from data_fetcher import get_nifty50_data
from stock_analyzer import format_stock_analysis
from config import NIFTY50_SYMBOLS, NIFTY500_SYMBOLS, POSITIONS
import os
from ai_analysis import conviction_gate, annotate_horizon_advice, generate_picks
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

def get_market_insights():
    """Get overall market insights and sample analysis for homepage."""
    market_data = get_nifty50_data(NIFTY50_SYMBOLS[:6])  # small sample for speed
    analysis = format_stock_analysis(market_data)

    insights = {
        'sentiment': 'Positive' if (sum(stock['health_score'] for stock in analysis) / max(1, len(analysis))) > 50 else 'Negative',
        'top_movers': len([stock for stock in analysis if abs(stock.get('day_change', 0)) > 2]),
        'buy_signals': len([stock for stock in analysis if stock['action'] == 'Buy']),
        'sell_signals': len([stock for stock in analysis if stock['action'] == 'Sell'])
    }
    return insights, analysis

def compute_kpis(analysis, positions):
    """Compute portfolio KPIs (value, today's pct) from holdings and analysis."""
    price_map = {s['symbol']: s for s in analysis}
    total_value = 0.0
    pnl_today = 0.0
    for p in positions:
        sym = p.get('symbol')
        qty = float(p.get('qty', 0) or 0)
        info = price_map.get(sym)
        if not info or qty <= 0:
            continue
        price = float(info.get('price', 0) or 0)
        change = float(info.get('price_change', 0) or 0)
        total_value += qty * price
        pnl_today += qty * change
    today_pct = (pnl_today / max(total_value - pnl_today, 1e-9) * 100) if total_value else 0.0
    return {'value': round(total_value, 2), 'today_pct': round(today_pct, 2)}

@app.route('/')
def index():
    """Render the main page with market insights and featured analysis."""
    insights, analysis = get_market_insights()
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

    return render_template('index.html', insights=insights, stocks=featured, kpi=kpi)

@app.route('/picks')
def picks_view():
    """Render Plutus Picks page with Buy/Sell lists for S/M/L terms."""
    # Use a larger universe if desired; here NIFTY50
    market_data = get_nifty50_data(NIFTY50_SYMBOLS)
    analysis = format_stock_analysis(market_data)
    scored = conviction_gate(analysis, min_score=0)
    scored = annotate_horizon_advice(scored)
    picks = generate_picks(scored, top_n=5)

    # KPIs for header
    try:
        pos_results = get_nifty50_data([p['symbol'] for p in POSITIONS])
        kpi = compute_kpis(format_stock_analysis(pos_results), POSITIONS)
    except Exception:
        kpi = {'value': 0.0, 'today_pct': 0.0}

    return render_template('picks.html', picks=picks, kpi=kpi)

@app.route('/api/picks')
def picks_api():
    """JSON API for Plutus Picks."""
    market_data = get_nifty50_data(NIFTY50_SYMBOLS)
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
    selected_symbols = request.json.get('symbols', [])
    if not selected_symbols:
        return jsonify({'error': 'Please select at least one stock'})
    
    # Get analysis results
    results = get_nifty50_data(selected_symbols)
    analysis = format_stock_analysis(results)
    
    return jsonify({'stocks': analysis})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
