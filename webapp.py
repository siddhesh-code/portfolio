"""
Flask web application for stock portfolio analysis
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from data_fetcher import get_nifty50_data
from stock_analyzer import format_stock_analysis
from config import NIFTY50_SYMBOLS
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Add a secret key for session management
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Enable template auto-reload

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
    """Get overall market insights"""
    # Get data for key market indicators
    market_data = get_nifty50_data(NIFTY50_SYMBOLS[:5])  # Using top 5 stocks as sample
    analysis = format_stock_analysis(market_data)

    if not analysis:
        return {
            'sentiment': 'Unknown',
            'top_movers': 0,
            'buy_signals': 0,
            'sell_signals': 0,
        }

    avg_health = sum(stock.get('health_score', 0) for stock in analysis) / len(analysis)

    # Calculate market insights
    insights = {
        'sentiment': 'Positive' if avg_health > 50 else 'Negative',
        'top_movers': len([stock for stock in analysis if abs(stock.get('day_change', 0)) > 2]),  # Stocks moving >2%
        'buy_signals': len([stock for stock in analysis if stock.get('action') == 'Buy']),
        'sell_signals': len([stock for stock in analysis if stock.get('action') == 'Sell'])
    }
    return insights

@app.route('/')
def index():
    """Render the main page with market insights"""
    insights = get_market_insights()
    return render_template('index.html', insights=insights)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze stock(s) from search"""
    symbols = request.form.get('symbols', '').strip().split(',')
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    
    if not symbols:
        return jsonify({'error': 'Please enter at least one stock symbol'})
    
    # Get analysis results
    results = get_nifty50_data(symbols)
    analysis = format_stock_analysis(results)
    
    return render_template('analysis.html', stocks=analysis)

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
