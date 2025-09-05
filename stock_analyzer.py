"""
Stock analysis and output formatting module
"""

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
        if data is None or data['technical'] is None:
            continue
            
        tech = data['technical'].iloc[-1]  # Get latest values
        
        try:
            # Calculate base values
            close_price = float(tech.get('Close', 0))
            open_price = float(tech.get('Open', close_price))
            
            # Calculate price changes
            price_change = close_price - open_price
            day_change = (price_change / open_price * 100) if open_price != 0 else 0
            
            # Set price targets
            price_target = float(tech.get('Price_Target', close_price * 1.05))
            stop_loss = float(tech.get('Stop_Loss', close_price * 0.95))
            
            # Calculate risk/reward ratio
            price_diff = price_target - close_price
            stop_diff = close_price - stop_loss
            risk_reward = min(abs(price_diff / stop_diff), 3.0) if stop_diff != 0 else 1.0
            
            # Technical analysis
            rsi = float(tech.get('RSI', 50))
            macd = float(tech.get('MACD', 0))
            macd_signal = float(tech.get('MACD_signal', 0))
            volume = int(tech.get('Volume', 0))
            volume_sma = float(tech.get('Volume_SMA', volume))
            
            # Format stock data
            stock_data = {
                'symbol': symbol,
                'price': close_price,
                'volume': volume,
                'health_score': int(tech.get('Health_Score', 50)),
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
                        'Resistance_1': close_price * 1.02,  # 2% above current price
                        'Resistance_2': close_price * 1.05,  # 5% above current price
                        'Support_1': close_price * 0.98,    # 2% below current price
                        'Support_2': close_price * 0.95     # 5% below current price
                    },
                    'BB': {
                        'Upper': float(tech.get('BB_upper', close_price * 1.02)),
                        'Lower': float(tech.get('BB_lower', close_price * 0.98)),
                        'Middle': float(tech.get('BB_middle', close_price)),
                        'Position': analyze_bollinger_bands(
                            close_price,
                            float(tech.get('BB_lower', close_price * 0.98)),
                            float(tech.get('BB_upper', close_price * 1.02))
                        )
                    },
                    'SMA': {
                        'SMA_20': float(tech.get('SMA_20', close_price)),
                        'SMA_50': float(tech.get('SMA_50', close_price)),
                        'Trend': analyze_moving_averages(
                            float(tech.get('SMA_20', close_price)),
                            float(tech.get('SMA_50', close_price))
                        )
                    },
                    'Volume': {
                        'Current': volume,
                        'SMA_20': volume_sma,
                        'Trend': analyze_volume(volume, volume_sma)
                    },
                    'Volatility': float(tech.get('Volatility', 0)) * 100  # Convert to percentage
                }
            }
            
            # Add backtest performance if available
            if any(col.startswith('Backtest_') for col in tech.index):
                stock_data['backtest'] = {
                    'total_return': float(tech.get('Backtest_total_return', 0)),
                    'annual_return': float(tech.get('Backtest_annual_return', 0)),
                    'sharpe_ratio': float(tech.get('Backtest_sharpe_ratio', 0)),
                    'max_drawdown': float(tech.get('Backtest_max_drawdown', 0))
                }
            
            formatted_results.append(stock_data)
            
        except (TypeError, ValueError, ZeroDivisionError) as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    return formatted_results
