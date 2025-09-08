"""
Stock analysis and output formatting module
"""

import math
import logging

# --- Tunables for neutral/dead-zone handling (kept as constants; no API changes) ---
_EPS_MACD_DIFF = 0.05           # MACD - Signal absolute diff below this => "Neutral"
_EPS_SMA_REL = 0.003            # 0.3% relative band around SMA cross => "Neutral"
_VOL_RATIO_LOW = 0.8            # Volume ratio bands for "Average"
_VOL_RATIO_HIGH = 1.2


def analyze_rsi(rsi):
    """Analyze RSI value and return signal"""
    if rsi < 30:
        return "Oversold"
    elif rsi > 70:
        return "Overbought"
    return "Neutral"


def analyze_macd(macd, signal):
    """Analyze MACD and return trend"""
    try:
        diff = float(macd) - float(signal)
        if abs(diff) < _EPS_MACD_DIFF:
            return "Neutral"
        return "Bullish" if diff > 0 else "Bearish"
    except Exception:
        return "Neutral"


def analyze_bollinger_bands(close, lower, upper):
    """Analyze position within Bollinger Bands"""
    try:
        upper = float(upper)
        lower = float(lower)
        close = float(close)
    except Exception:
        return 50.0, "Neutral"

    if upper == lower:
        return 50.0, "Neutral"

    # position 0..100, clamped
    position = (close - lower) / (upper - lower) * 100.0
    position = max(0.0, min(100.0, position))
    signal = "Oversold" if position < 20 else "Overbought" if position > 80 else "Neutral"
    return position, signal


def analyze_moving_averages(sma20, sma50):
    """Analyze moving averages crossover"""
    try:
        s20 = float(sma20)
        s50 = float(sma50)
        # Neutral band: if relative gap is tiny, treat as "Neutral"
        base = abs(s50) if s50 != 0 else max(abs(s20), 1.0)
        if base > 0 and abs(s20 - s50) / base < _EPS_SMA_REL:
            return "Neutral"
        return "Bullish" if s20 > s50 else "Bearish"
    except Exception:
        return "Neutral"


def analyze_volume(current, average):
    """Analyze volume compared to average"""
    try:
        current = float(current)
        average = float(average)
        if average <= 0:
            return "Neutral"
        ratio = current / average
        if ratio < _VOL_RATIO_LOW:
            return "Below Average"
        if ratio > _VOL_RATIO_HIGH:
            return "Above Average"
        return "Average"
    except Exception:
        return "Neutral"


def format_stock_analysis(results):
    """Format the analysis output for the stocks"""
    formatted_results = []

    for symbol, data in results.items():
        if data is None or data['technical'] is None:
            continue

        dft = data['technical']
        if len(dft) == 0:
            continue

        tech = dft.iloc[-1]  # Get latest values

        try:
            # Calculate base values (robust conversions)
            def _f(x, default=0.0):
                try:
                    v = float(x)
                    if math.isnan(v):
                        return float(default)
                    return v
                except Exception:
                    return float(default)

            def _i(x, default=0):
                try:
                    v = float(x)
                    if math.isnan(v):
                        return int(default)
                    return int(v)
                except Exception:
                    return int(default)

            close_price = _f(tech.get('Close', 0), 0.0)
            open_price = _f(tech.get('Open', close_price), close_price)

            # Calculate price changes
            price_change = close_price - open_price
            day_change = (price_change / open_price * 100.0) if open_price != 0 else 0.0  # percent

            # Set price targets
            price_target = _f(tech.get('Price_Target', close_price * 1.05), close_price * 1.05)
            stop_loss = _f(tech.get('Stop_Loss', close_price * 0.95), close_price * 0.95)

            # Calculate risk/reward ratio (cap at 3.0; guard divide-by-zero/negative stop diff)
            price_diff = price_target - close_price
            stop_diff = close_price - stop_loss
            if stop_diff > 0:
                risk_reward = min(abs(price_diff / stop_diff), 3.0)
            else:
                risk_reward = 1.0  # conservative default

            # Technical analysis inputs
            rsi = _f(tech.get('RSI', 50), 50)
            macd = _f(tech.get('MACD', 0), 0)
            macd_signal = _f(tech.get('MACD_signal', 0), 0)
            volume_current = _i(tech.get('Volume', 0), 0)
            volume_sma = _f(tech.get('Volume_SMA', volume_current), volume_current)

            # Compute multi-horizon metrics and chart series (last 90 points)
            try:
                tail = dft.tail(90).copy()

                def _ts_to_ms(ts):
                    # Robust pandas/numpy datetime -> ms since epoch
                    try:
                        return int(ts.value // 1_000_000)  # pandas.Timestamp in ns
                    except Exception:
                        try:
                            return int(ts.astype('datetime64[ms]').astype('int64'))
                        except Exception:
                            return None

                # Build aligned time array
                times = []
                for idx in tail.index:
                    t = _ts_to_ms(idx)
                    if t is not None:
                        times.append(t)
                    else:
                        # keep placeholder to maintain alignment
                        times.append(None)

                # Helpers to build XY series safely
                def series_xy(col):
                    if col not in tail.columns:
                        return []
                    arr = tail[col]
                    out = []
                    n = len(arr)
                    for i in range(n):
                        t = times[i]
                        if t is None:
                            continue
                        try:
                            val = float(arr.iloc[i])
                        except Exception:
                            val = None
                        out.append({'x': t, 'y': val})
                    return out

                # Candle series (OHLC)
                o = tail.get('Open', None)
                h = tail.get('High', None)
                l = tail.get('Low', None)
                c = tail.get('Close', None)
                candles = []
                if o is not None and h is not None and l is not None and c is not None:
                    n = min(len(times), len(o), len(h), len(l), len(c))
                    for i in range(n):
                        t = times[i]
                        if t is None:
                            continue
                        try:
                            candles.append({
                                'x': t,
                                'y': [float(o.iloc[i]), float(h.iloc[i]), float(l.iloc[i]), float(c.iloc[i])]
                            })
                        except Exception:
                            continue

                # Volume series
                volume_series = series_xy('Volume')

                # Momentum returns (percent values for consistency)
                try:
                    def t_ret_pct(n):
                        if len(tail) > (n + 1):
                            prev = float(tail['Close'].iloc[-n-1])
                            last = float(tail['Close'].iloc[-1])
                            if prev != 0:
                                return (last / prev - 1.0) * 100.0
                        return 0.0

                    ret_5d = t_ret_pct(5)
                    ret_21d = t_ret_pct(21)   # ~1 month
                    ret_63d = t_ret_pct(63)   # ~3 months
                except Exception:
                    ret_5d = ret_21d = ret_63d = 0.0

                # MACD histogram slope (last diff)
                try:
                    mhd = dft.get('MACD_histogram', None) or dft.get('MACD_hist', None)
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
                    's1': _f(tech.get('Support_1') or 0, 0),
                    's2': _f(tech.get('Support_2') or 0, 0),
                    'r1': _f(tech.get('Resistance_1') or 0, 0),
                    'r2': _f(tech.get('Resistance_2') or 0, 0)
                }
            except Exception:
                chart = None
                ret_5d = ret_21d = ret_63d = 0.0
                macd_hist_slope = 0.0

            # Derive display fields
            info = data.get('info', {}) if isinstance(data, dict) else {}
            display_symbol = symbol.split('.')[0] if isinstance(symbol, str) else symbol
            name = info.get('shortName') or info.get('longName') or display_symbol

            # Compute Bollinger metrics once for embedding
            bb_pos, bb_sig = analyze_bollinger_bands(
                close_price,
                _f(tech.get('BB_lower', close_price * 0.98), close_price * 0.98),
                _f(tech.get('BB_upper', close_price * 1.02), close_price * 1.02)
            )

            # Format stock data
            stock_data = {
                'symbol': symbol,
                'display_symbol': display_symbol,
                'name': name,
                'price': close_price,
                'volume': volume_current,
                'health_score': int(_f(tech.get('Health_Score', 50), 50)),
                'trend': str(tech.get('Trend', 'Sideways')),
                'trend_strength': str(tech.get('Trend_Strength', 'Neutral')),
                'action': str(tech.get('Action', 'Watch')),
                'action_reason': str(tech.get('Action_Reason', 'Insufficient data')),
                'price_change': price_change,
                'day_change': day_change,  # percent
                'price_target': price_target,
                'stop_loss': stop_loss,
                'risk_reward': risk_reward,
                'metrics': {
                    'ret_5d': ret_5d,
                    'ret_21d': ret_21d,
                    'ret_63d': ret_63d,
                    'macd_hist_slope': macd_hist_slope,
                    'dist_r1': ((_f(tech.get('Resistance_1', 0), 0) - close_price) / close_price) if _f(tech.get('Resistance_1', 0), 0) > 0 else 0.0,
                    'dist_s1': ((close_price - _f(tech.get('Support_1', 0), 0)) / close_price) if _f(tech.get('Support_1', 0), 0) > 0 else 0.0,
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
                        'Resistance_1': _f(tech.get('Resistance_1', close_price * 1.02), close_price * 1.02),
                        'Resistance_2': _f(tech.get('Resistance_2', close_price * 1.05), close_price * 1.05),
                        'Support_1': _f(tech.get('Support_1', close_price * 0.98), close_price * 0.98),
                        'Support_2': _f(tech.get('Support_2', close_price * 0.95), close_price * 0.95)
                    },
                    'BB': {
                        'Upper': _f(tech.get('BB_upper', close_price * 1.02), close_price * 1.02),
                        'Lower': _f(tech.get('BB_lower', close_price * 0.98), close_price * 0.98),
                        'Middle': _f(tech.get('BB_middle', close_price), close_price),
                        'Position': (bb_pos, bb_sig)  # preserve (position, signal) tuple
                    },
                    'SMA': {
                        'SMA_20': _f(tech.get('SMA_20', close_price), close_price),
                        'SMA_50': _f(tech.get('SMA_50', close_price), close_price),
                        'Trend': analyze_moving_averages(
                            _f(tech.get('SMA_20', close_price), close_price),
                            _f(tech.get('SMA_50', close_price), close_price)
                        )
                    },
                    'Volume': {
                        'Current': volume_current,
                        'SMA_20': volume_sma,
                        'Trend': analyze_volume(volume_current, volume_sma)
                    },
                    'Volatility': _f(tech.get('Volatility', 0), 0) * 100.0  # percent
                },
                'chart': chart
            }

            # Add backtest performance if available
            try:
                if any(str(col).startswith('Backtest_') for col in tech.index):
                    stock_data['backtest'] = {
                        'total_return': _f(tech.get('Backtest_total_return', 0), 0),
                        'annual_return': _f(tech.get('Backtest_annual_return', 0), 0),
                        'sharpe_ratio': _f(tech.get('Backtest_sharpe_ratio', 0), 0),
                        'max_drawdown': _f(tech.get('Backtest_max_drawdown', 0), 0)
                    }
            except Exception:
                pass

            formatted_results.append(stock_data)

        except (TypeError, ValueError, ZeroDivisionError) as e:
            logging.warning(f"Error processing {symbol}: {str(e)}")
            continue

    return formatted_results
