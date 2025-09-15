"""
Stock analysis and output formatting module
"""

import math


def _safe_float(value, default=0.0):
    """Convert *value* to ``float`` while guarding against NaNs and ``None``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _safe_int(value, default=0):
    """Convert *value* to ``int`` with graceful fallback for bad inputs."""
    try:
        result = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default

    return result


def analyze_rsi(rsi):
    """Analyze RSI value and return signal"""
    if rsi < 30:
        return "Oversold"
    elif rsi > 70:
        return "Overbought"
    return "Neutral"

def analyze_macd(macd, signal):
    """Analyze MACD and return trend"""
    return "Bullish" if macd > signal else "Bearish"

def analyze_bollinger_bands(close, lower, upper):
    """Analyze position within Bollinger Bands"""
    if upper == lower:
        return 50, "Neutral"
    position = (close - lower) / (upper - lower) * 100
    signal = "Oversold" if position < 20 else "Overbought" if position > 80 else "Neutral"
    return position, signal

def analyze_moving_averages(sma20, sma50):
    """Analyze moving averages crossover"""
    return "Bullish" if sma20 > sma50 else "Bearish"

def analyze_volume(current, average):
    """Analyze volume compared to average"""
    return "Above Average" if current > average else "Below Average"

def format_stock_analysis(results):
    """Format the analysis output for the stocks"""
    formatted_results = []
    
    for symbol, data in results.items():
        if not data or data.get('technical') is None:
            continue

        technical = data['technical']
        if technical is None or technical.empty:
            continue

        tech = technical.iloc[-1]  # Get latest values

        try:
            # Calculate base values
            close_price = _safe_float(tech.get('Close'), 0.0)
            open_price = _safe_float(tech.get('Open'), close_price)

            if close_price <= 0:
                continue

            # Calculate price changes
            price_change = close_price - open_price
            day_change = (price_change / open_price * 100) if open_price else 0.0

            # Set price targets
            default_target = close_price * 1.05
            default_stop = close_price * 0.95
            price_target = _safe_float(tech.get('Price_Target'), default_target)
            stop_loss = _safe_float(tech.get('Stop_Loss'), default_stop)

            # Calculate risk/reward ratio
            risk = max(close_price - stop_loss, 0.0)
            reward = max(price_target - close_price, 0.0)
            risk_reward = 0.0
            if risk > 0 and reward > 0:
                risk_reward = min(reward / risk, 3.0)

            # Technical analysis
            rsi = _safe_float(tech.get('RSI'), 50.0)
            macd = _safe_float(tech.get('MACD'), 0.0)
            macd_signal = _safe_float(tech.get('MACD_signal'), 0.0)
            volume = _safe_int(tech.get('Volume'), 0)
            volume_sma = _safe_float(tech.get('Volume_SMA'), float(volume))

            support_1 = _safe_float(tech.get('Support_1'), close_price * 0.98)
            support_2 = _safe_float(tech.get('Support_2'), close_price * 0.95)
            resistance_1 = _safe_float(tech.get('Resistance_1'), close_price * 1.02)
            resistance_2 = _safe_float(tech.get('Resistance_2'), close_price * 1.05)

            bb_upper = _safe_float(tech.get('BB_upper'), close_price * 1.02)
            bb_lower = _safe_float(tech.get('BB_lower'), close_price * 0.98)
            bb_middle = _safe_float(tech.get('BB_middle'), close_price)
            bb_position = analyze_bollinger_bands(close_price, bb_lower, bb_upper)

            sma_20 = _safe_float(tech.get('SMA_20'), close_price)
            sma_50 = _safe_float(tech.get('SMA_50'), close_price)

            # Format stock data
            stock_data = {
                'symbol': symbol,
                'price': close_price,
                'volume': volume,
                'health_score': _safe_int(tech.get('Health_Score'), 50),
                'trend': str(tech.get('Trend', 'Sideways')),
                'trend_strength': str(tech.get('Trend_Strength', 'Neutral')),
                'action': str(tech.get('Action', 'Watch')),
                'action_reason': str(tech.get('Action_Reason', 'Insufficient data')),
                'price_change': price_change,
                'day_change': day_change,
                'price_target': price_target,
                'stop_loss': stop_loss,
                'risk_reward': risk_reward,
                'indicators': {
                    'RSI': {
                        'value': rsi,
                        'signal': analyze_rsi(rsi)
                    },
                    'MACD': {
                        'MACD': macd,
                        'Signal': macd_signal,
                        'Histogram': macd - macd_signal,
                        'Trend': analyze_macd(macd, macd_signal)
                    },
                    'Support_Resistance': {
                        'Resistance_1': resistance_1,
                        'Resistance_2': resistance_2,
                        'Support_1': support_1,
                        'Support_2': support_2
                    },
                    'BB': {
                        'Upper': bb_upper,
                        'Lower': bb_lower,
                        'Middle': bb_middle,
                        'Position': bb_position
                    },
                    'SMA': {
                        'SMA_20': sma_20,
                        'SMA_50': sma_50,
                        'Trend': analyze_moving_averages(
                            sma_20,
                            sma_50
                        )
                    },
                    'Volume': {
                        'Current': volume,
                        'SMA_20': volume_sma,
                        'Trend': analyze_volume(volume, volume_sma)
                    },
                    'Volatility': _safe_float(tech.get('Volatility'), 0.0) * 100  # Convert to percentage
                }
            }

            # Add backtest performance if available
            if any(col.startswith('Backtest_') for col in tech.index):
                stock_data['backtest'] = {
                    'total_return': _safe_float(tech.get('Backtest_total_return'), 0.0),
                    'annual_return': _safe_float(tech.get('Backtest_annual_return'), 0.0),
                    'sharpe_ratio': _safe_float(tech.get('Backtest_sharpe_ratio'), 0.0),
                    'max_drawdown': _safe_float(tech.get('Backtest_max_drawdown'), 0.0)
                }

            formatted_results.append(stock_data)

        except (TypeError, ValueError, ZeroDivisionError) as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    return formatted_results
