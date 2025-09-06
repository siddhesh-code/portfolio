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
            
        dft = data['technical']
        tech = dft.iloc[-1]  # Get latest values
        
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
            volume_current = int(tech.get('Volume', 0))
            volume_sma = float(tech.get('Volume_SMA', volume_current))
            
            # Compute multi-horizon metrics and chart series (last 90 points)
            try:
                tail = dft.tail(90).copy()
                # epoch ms for ApexCharts
                def to_ms(idx):
                    try:
                        return int(getattr(idx, 'to_datetime64')().astype('datetime64[ms]').astype('int64'))
                    except Exception:
                        try:
                            # pandas Timestamp
                            return int(idx.value // 1_000_000)
                        except Exception:
                            return None

                times = [to_ms(i) for i in tail.index]
                o = tail.get('Open', [])
                h = tail.get('High', [])
                l = tail.get('Low', [])
                c = tail.get('Close', [])
                candles = []
                for i, t in enumerate(times):
                    if t is None:
                        continue
                    try:
                        candles.append({
                            'x': t,
                            'y': [float(o.iloc[i]), float(h.iloc[i]), float(l.iloc[i]), float(c.iloc[i])]
                        })
                    except Exception:
                        continue

                def series_xy(col):
                    arr = tail.get(col, [])
                    out = []
                    for i, t in enumerate(times):
                        try:
                            val = float(arr.iloc[i])
                        except Exception:
                            val = None
                        out.append({'x': times[i], 'y': val})
                    return out

                # Volume series
                vol = tail.get('Volume', [])
                volume_series = []
                for i, t in enumerate(times):
                    try:
                        v = float(vol.iloc[i])
                    except Exception:
                        v = 0.0
                    volume_series.append({'x': times[i], 'y': v})

                # Momentum returns
                try:
                    # Use business-day approximations: 5d, 21d (~1m), 63d (~3m)
                    def t_ret(n):
                        if len(tail) > n and float(tail['Close'].iloc[-n-1]) != 0:
                            return float(tail['Close'].iloc[-1] / tail['Close'].iloc[-n-1] - 1)
                        return 0.0
                    ret_5d = t_ret(5)
                    ret_21d = t_ret(21)
                    ret_63d = t_ret(63)
                except Exception:
                    ret_5d = ret_21d = ret_63d = 0.0

                # MACD histogram slope (last diff)
                try:
                    mhd = dft.get('MACD_histogram', None)
                    if mhd is not None and len(mhd) >= 2:
                        macd_hist_slope = float(mhd.iloc[-1] - mhd.iloc[-2])
                    else:
                        macd_hist_slope = 0.0
                except Exception:
                    macd_hist_slope = 0.0

                chart = {
                    'candles': candles,
                    'sma20': series_xy('SMA_20'),
                    'bb_upper': series_xy('BB_upper'),
                    'bb_lower': series_xy('BB_lower'),
                    'volume': volume_series,
                    's1': float(tech.get('Support_1') or 0),
                    's2': float(tech.get('Support_2') or 0),
                    'r1': float(tech.get('Resistance_1') or 0),
                    'r2': float(tech.get('Resistance_2') or 0)
                }
            except Exception:
                chart = None

            # Derive display fields
            info = data.get('info', {}) if isinstance(data, dict) else {}
            display_symbol = symbol.split('.')[0] if isinstance(symbol, str) else symbol
            name = info.get('shortName') or info.get('longName') or display_symbol

            # Format stock data
            stock_data = {
                'symbol': symbol,
                'display_symbol': display_symbol,
                'name': name,
                'price': close_price,
                'volume': volume_current,
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
                'metrics': {
                    'ret_5d': ret_5d if 'ret_5d' in locals() else 0.0,
                    'ret_21d': ret_21d if 'ret_21d' in locals() else 0.0,
                    'ret_63d': ret_63d if 'ret_63d' in locals() else 0.0,
                    'macd_hist_slope': macd_hist_slope if 'macd_hist_slope' in locals() else 0.0,
                    'dist_r1': ((float(tech.get('Resistance_1', 0)) - close_price) / close_price) if float(tech.get('Resistance_1', 0) or 0) > 0 else 0.0,
                    'dist_s1': ((close_price - float(tech.get('Support_1', 0))) / close_price) if float(tech.get('Support_1', 0) or 0) > 0 else 0.0,
                },
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
                        # Prefer technically derived levels from indicators.py, fallback to percentage bands
                        'Resistance_1': float(tech.get('Resistance_1', close_price * 1.02)),
                        'Resistance_2': float(tech.get('Resistance_2', close_price * 1.05)),
                        'Support_1': float(tech.get('Support_1', close_price * 0.98)),
                        'Support_2': float(tech.get('Support_2', close_price * 0.95))
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
                        'Current': volume_current,
                        'SMA_20': volume_sma,
                        'Trend': analyze_volume(volume_current, volume_sma)
                    },
                    'Volatility': float(tech.get('Volatility', 0)) * 100  # Convert to percentage
                },
                'chart': chart
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
