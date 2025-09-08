"""
Technical indicators calculation module with time series analysis and trading signals
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, AccDistIndexIndicator
from scipy.signal import argrelextrema
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, VOLUME_PERIOD, VOLATILITY_PERIOD,
    SMA_PERIODS, EMA_PERIOD
)

# --- Internal constants (business-logic tunings; no API changes) ---
_MIN_TOL = 1e-3                  # minimum price tolerance to avoid zero/NaN issues
_MAX_LEVELS = 5                  # cap for support/resistance points
_ADX_UP = 25
_ADX_MID = 18
_RSI_PCT_LOW = 0.15              # dynamic RSI lower quantile
_RSI_PCT_HIGH = 0.85             # dynamic RSI upper quantile
_VOL_Z_SPIKE = 2.0
_COST_BPS = 0.0005
_SLIP_BPS = 0.0005


# --------------------------- Support / Resistance ----------------------------

def identify_support_resistance(df, window=20, num_points=5):
    """
    Identify support and resistance using local extrema and ATR-based clustering.
    Returns (support_levels, resistance_levels) lists (ascending for supports, descending for resistances).
    """
    if df.empty or len(df) < max(window * 3, 40):
        return [], []

    # Local extrema
    local_max_idx = argrelextrema(df['High'].values, np.greater, order=window)[0]
    local_min_idx = argrelextrema(df['Low'].values, np.less, order=window)[0]
    highs = pd.Series(df['High'].values[local_max_idx], index=df.index[local_max_idx])
    lows = pd.Series(df['Low'].values[local_min_idx], index=df.index[local_min_idx])

    # ATR tolerance
    try:
        atr_series = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True).average_true_range()
        tol = float(atr_series.tail(window).mean())
    except Exception:
        tol = float((df['High'] - df['Low']).tail(20).mean())
    if not np.isfinite(tol) or tol <= 0:
        tol = _MIN_TOL

    # recency helper
    end_ts = df.index[-1]
    try:
        end_ts = pd.Timestamp(end_ts)
    except Exception:
        pass

    def _recency_weight(ts):
        try:
            ts = pd.Timestamp(ts)
            dt = end_ts - ts
            secs = abs(getattr(dt, "total_seconds", lambda: 0.0)())
            # half-life ~ 60 trading days -> gentle decay
            return 1.0 / (1.0 + secs / (60 * 24 * 3600))
        except Exception:
            return 1.0

    def _cluster_levels(series, is_res):
        if series.empty:
            return []
        lvls = series.sort_values(ascending=not is_res).copy()
        clusters = []
        for ts, price in lvls.items():
            price = float(price)
            placed = False
            for c in clusters:
                if abs(price - c['price']) <= tol:
                    c['count'] += 1
                    w = _recency_weight(ts)
                    c['score'] += 1.0 + w
                    # weighted centroid update
                    c['price'] = (c['price'] * (c['count'] - 1) + price) / c['count']
                    placed = True
                    break
            if not placed:
                clusters.append({'price': float(price), 'count': 1, 'score': 1.0})
        clusters.sort(key=lambda x: (x['score'], x['count']), reverse=True)
        return [round(c['price'], 6) for c in clusters[: max(num_points * 2, _MAX_LEVELS)]]

    res_candidates = _cluster_levels(highs, is_res=True)
    sup_candidates = _cluster_levels(lows, is_res=False)

    def _dedupe(levels, is_res):
        out = []
        for p in (sorted(levels, reverse=is_res)):
            if not out or all(abs(p - q) > tol * 0.75 for q in out):
                out.append(p)
            if len(out) >= num_points:
                break
        return out

    resistance_levels = _dedupe(res_candidates, is_res=True)
    support_levels = _dedupe(sup_candidates, is_res=False)

    resistance_levels = sorted(resistance_levels, reverse=True)
    support_levels = sorted(support_levels)
    return support_levels, resistance_levels


# ------------------------------ Price Patterns ------------------------------

def analyze_price_patterns(df, window=20):
    """
    Identify simple patterns with ATR-aware tolerances.
    Returns dict with: double_top, double_bottom, head_shoulders, trend.
    """
    patterns = {
        'double_top': False,
        'double_bottom': False,
        'head_shoulders': False,
        'trend': 'neutral'
    }
    if df.empty or len(df) < max(window * 3, 60):
        return patterns

    # Trend via SMA slope + ADX
    try:
        sma50 = df['SMA_50']
        slope50 = float(sma50.iloc[-1] - sma50.iloc[-window])
        adx_last = float(ADXIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True).adx().iloc[-1])
        if slope50 > 0 and adx_last >= _ADX_MID:
            patterns['trend'] = 'uptrend'
        elif slope50 < 0 and adx_last >= _ADX_MID:
            patterns['trend'] = 'downtrend'
        else:
            patterns['trend'] = 'neutral'
    except Exception:
        try:
            patterns['trend'] = 'uptrend' if df['SMA_50'].iloc[-1] > df['SMA_50'].iloc[-window] else \
                                'downtrend' if df['SMA_50'].iloc[-1] < df['SMA_50'].iloc[-window] else 'neutral'
        except Exception:
            patterns['trend'] = 'neutral'

    # ATR tolerance
    try:
        atr_last = float(AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True).average_true_range().iloc[-1])
    except Exception:
        atr_last = float((df['High'] - df['Low']).tail(20).mean())
    tol = max(0.5 * atr_last, 0.01 * float(df['Close'].iloc[-1]), _MIN_TOL)

    # Peaks / troughs
    highs_idx = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows_idx = argrelextrema(df['Low'].values, np.less, order=window)[0]

    # Double top / bottom
    if len(highs_idx) >= 2:
        h1, h2 = df['High'].iloc[highs_idx[-2]], df['High'].iloc[highs_idx[-1]]
        if abs(h1 - h2) <= tol and highs_idx[-1] - highs_idx[-2] >= max(2, window // 2):
            mid_low = df['Low'].iloc[highs_idx[-2]:highs_idx[-1]].min()
            if mid_low + tol < min(h1, h2):
                patterns['double_top'] = True

    if len(lows_idx) >= 2:
        l1, l2 = df['Low'].iloc[lows_idx[-2]], df['Low'].iloc[lows_idx[-1]]
        if abs(l1 - l2) <= tol and lows_idx[-1] - lows_idx[-2] >= max(2, window // 2):
            mid_high = df['High'].iloc[lows_idx[-2]:lows_idx[-1]].max()
            if mid_high - tol > max(l1, l2):
                patterns['double_bottom'] = True

    # Head & Shoulders (very heuristic)
    if len(highs_idx) >= 3:
        p1, p2, p3 = highs_idx[-3], highs_idx[-2], highs_idx[-1]
        h1, h2, h3 = df['High'].iloc[p1], df['High'].iloc[p2], df['High'].iloc[p3]
        shoulders_close = abs(h1 - h3) <= tol
        middle_higher = (h2 - max(h1, h3)) >= 0.5 * tol
        spacing_ok = (p3 - p2) >= max(2, window // 4) and (p2 - p1) >= max(2, window // 4)
        patterns['head_shoulders'] = bool(shoulders_close and middle_higher and spacing_ok)

    return patterns


# --------------------------------- Signals ----------------------------------

def generate_trading_signals(df):
    """
    Generate trading signals: 1 (buy), -1 (sell), 0 (hold)
    Uses dynamic RSI bands, MACD cross quality, BB bounce, MA stack + slope, volume z-score,
    and light hysteresis to reduce churn.
    """
    signals = pd.DataFrame(index=df.index)

    # Dynamic RSI bands using recent window quantiles (fallback 30/70)
    rsi_series = df['RSI'].astype(float).dropna()
    if len(rsi_series) >= 100:
        window = 80
        rsi_tail = rsi_series.tail(window).values
        rsi_low = float(np.quantile(rsi_tail, _RSI_PCT_LOW))
        rsi_high = float(np.quantile(rsi_tail, _RSI_PCT_HIGH))
    else:
        rsi_low, rsi_high = 30.0, 70.0

    # RSI signal
    signals['rsi_signal'] = 0
    with np.errstate(invalid='ignore'):
        signals.loc[df['RSI'] <= rsi_low, 'rsi_signal'] = 1
        signals.loc[df['RSI'] >= rsi_high, 'rsi_signal'] = -1

    # MACD signal: require side AND improving/falling histogram
    hist = (df['MACD'] - df['MACD_signal']).astype(float)
    hist_slope = hist.diff()
    macd_up = (df['MACD'] > df['MACD_signal']) & (hist_slope > 0)
    macd_dn = (df['MACD'] < df['MACD_signal']) & (hist_slope < 0)
    signals['macd_signal'] = 0
    signals.loc[macd_up.fillna(False), 'macd_signal'] = 1
    signals.loc[macd_dn.fillna(False), 'macd_signal'] = -1

    # Bollinger: re-enter bands
    prev_lower = df['BB_lower'].shift(1)
    prev_upper = df['BB_upper'].shift(1)
    reenter_lower = (df['Close'] > df['BB_lower']) & (df['Close'].shift(1) < prev_lower)
    reenter_upper = (df['Close'] < df['BB_upper']) & (df['Close'].shift(1) > prev_upper)
    signals['bb_signal'] = 0
    signals.loc[reenter_lower.fillna(False), 'bb_signal'] = 1
    signals.loc[reenter_upper.fillna(False), 'bb_signal'] = -1

    # MA signal: stack + slope
    sma20 = df['SMA_20'].astype(float)
    sma50 = df['SMA_50'].astype(float)
    slope20 = sma20.diff()
    slope50 = sma50.diff()
    signals['ma_signal'] = 0
    signals.loc[(sma20 > sma50) & (slope20 > 0) & (slope50 >= 0), 'ma_signal'] = 1
    signals.loc[(sma20 < sma50) & (slope20 < 0) & (slope50 <= 0), 'ma_signal'] = -1

    # Volume: z-score vs rolling mean/std plus Volume_SMA fallback
    vol = df['Volume'].astype(float)
    vol_mu = vol.rolling(VOLUME_PERIOD).mean()
    vol_sd = vol.rolling(VOLUME_PERIOD).std(ddof=0).replace(0, np.nan)
    vol_z = (vol - vol_mu) / vol_sd
    vol_spike = (vol_z >= _VOL_Z_SPIKE) | (vol > df['Volume_SMA'])
    signals['volume_signal'] = 0
    signals.loc[vol_spike.fillna(False), 'volume_signal'] = 1

    # Blend weights (unchanged keys, slight tuning ok)
    weights = {
        'rsi_signal': 0.2,
        'macd_signal': 0.3,
        'bb_signal': 0.2,
        'ma_signal': 0.2,
        'volume_signal': 0.1
    }
    signals['combined_signal'] = sum(signals[col].astype(float).reindex(df.index).fillna(0.0) * w
                                     for col, w in weights.items())

    # Position with light hysteresis
    upper, lower = 0.35, -0.35
    signals['position'] = 0
    signals.loc[signals['combined_signal'] > upper, 'position'] = 1
    signals.loc[signals['combined_signal'] < lower, 'position'] = -1
    signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0).astype(float)

    return signals


# -------------------------------- Backtest ----------------------------------

def backtest_strategy(df, signals, initial_capital=100000.0):
    """
    Backtest the strategy with T+1 execution, simple costs & slippage.
    Returns (portfolio_df, performance_dict).
    """
    if df.empty or signals.empty:
        return pd.DataFrame(index=df.index), {
            'total_return': 0.0, 'annual_return': 0.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0
        }

    price = df['Close'].astype(float)
    pos = signals['position'].astype(float)

    # Execute at next bar's close (avoid look-ahead)
    pos_exec = pos.shift(1).fillna(0.0)

    # Friction on turnover
    turnover = pos_exec.diff().abs().fillna(0.0)
    costs = (_COST_BPS + _SLIP_BPS) * turnover

    # Strategy returns (daily), then subtract costs
    strat_ret = pos_exec * price.pct_change().fillna(0.0) - costs

    # Equity curve (self-consistent)
    total = initial_capital * (1.0 + strat_ret).replace([np.inf, -np.inf], 0.0).cumprod()

    portfolio = pd.DataFrame(index=signals.index)
    portfolio['returns'] = strat_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    portfolio['total'] = total.ffill()
    # For reporting, split into "holdings" and "cash" (notional)
    portfolio['holdings'] = pos_exec * price
    portfolio['cash'] = portfolio['total'] - portfolio['holdings']

    # Performance stats
    total_return = float(portfolio['total'].iloc[-1] / max(initial_capital, 1e-9) - 1.0)
    mean_ret = float(portfolio['returns'].mean())
    std_ret = float(portfolio['returns'].std(ddof=0) or 0.0)
    annual_return = mean_ret * 252.0
    sharpe = (mean_ret / std_ret) * np.sqrt(252.0) if std_ret > 0 else 0.0
    cummax = portfolio['total'].cummax()
    max_dd = float((portfolio['total'] / cummax - 1.0).min())

    performance = {
        'total_return': total_return,
        'annual_return': float(annual_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd)
    }
    return portfolio, performance


# ------------------------------ Health / Trend ------------------------------

def calculate_health_score(df):
    """Calculate stock health score (0-100) with trend, momentum, volume, volatility."""
    if df.empty:
        return 0
    latest = df.iloc[-1]
    score = 0

    # Trend (30)
    sma50 = latest.get('SMA_50', np.nan)
    sma200 = latest.get('SMA_200', np.nan)
    close = latest.get('Close', np.nan)
    if np.isfinite(sma50) and np.isfinite(close):
        if np.isfinite(sma200):
            if sma50 > sma200:
                score += 15
                if close > sma50:
                    score += 15
        else:
            # no SMA200 -> lighter weight
            if close > sma50:
                score += 20

    # Volume (20)
    if latest.get('Volume', 0) > 0:
        vol_tail = df['Volume'].tail(VOLUME_PERIOD).astype(float)
        vol_mu = float(vol_tail.mean())
        vol_sd = float(vol_tail.std(ddof=0) or 1.0)
        vol_z = (float(latest['Volume']) - vol_mu) / vol_sd
        score += 20 if vol_z >= 1.5 else 10 if latest.get('Volume_SMA', 0) and latest['Volume'] > latest['Volume_SMA'] else 0

    # Momentum (30)
    if np.isfinite(latest.get('RSI', np.nan)) and latest['RSI'] > 55:
        score += 15
    if np.isfinite(latest.get('MACD', np.nan)) and np.isfinite(latest.get('MACD_signal', np.nan)) and latest['MACD'] > latest['MACD_signal']:
        score += 15

    # Volatility (20) – tighter bands preferred
    bb_mid = latest.get('BB_middle', np.nan)
    bb_up = latest.get('BB_upper', np.nan)
    bb_low = latest.get('BB_lower', np.nan)
    if np.isfinite(bb_mid) and bb_mid != 0 and np.isfinite(bb_up) and np.isfinite(bb_low):
        bb_width = (bb_up - bb_low) / abs(bb_mid)
        score += 20 if bb_width < 0.10 else 10 if bb_width < 0.20 else 0

    return int(min(100, max(0, score)))


def determine_trend(df):
    """Determine trend direction and strength using SMA stack and ADX if available."""
    latest = df.iloc[-1]
    sma50 = latest.get('SMA_50', np.nan)
    sma200 = latest.get('SMA_200', np.nan)
    close = latest.get('Close', np.nan)

    direction = "Sideways"
    if np.isfinite(close) and np.isfinite(sma50):
        if (close > sma50) and (not np.isfinite(sma200) or sma50 > sma200):
            direction = "Uptrend"
        elif (close < sma50) and (not np.isfinite(sma200) or sma50 < sma200):
            direction = "Downtrend"

    try:
        adx_last = float(ADXIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True).adx().iloc[-1])
    except Exception:
        adx_last = np.nan

    strength = "Weak"
    if np.isfinite(sma200) and sma200 != 0 and np.isfinite(close):
        dist = abs(close - sma200) / abs(sma200)
        if dist > 0.05 or (np.isfinite(adx_last) and adx_last >= _ADX_UP):
            strength = "Strong"
        elif dist > 0.02 or (np.isfinite(adx_last) and adx_last >= _ADX_MID):
            strength = "Moderate"
    else:
        strength = "Moderate" if (np.isfinite(adx_last) and adx_last >= _ADX_MID) else "Weak"

    return direction, strength


# ----------------------------- Actions / Targets ----------------------------

def get_action_and_reason(df):
    """Determine trading action and provide reason using ATR & volume confirmation."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    # Recent range & ATR
    recent_high = float(df['High'].tail(20).max())
    recent_low = float(df['Low'].tail(20).min())
    try:
        atr_last = float(AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True).average_true_range().iloc[-1])
    except Exception:
        atr_last = float((df['High'] - df['Low']).tail(14).mean())
    atr_last = max(atr_last, _MIN_TOL)

    vol_tail = df['Volume'].tail(VOLUME_PERIOD).astype(float)
    vol_mu = float(vol_tail.mean())
    vol_sd = float(vol_tail.std(ddof=0) or 1.0)
    vol_z = (float(latest.get('Volume', 0.0)) - vol_mu) / vol_sd

    # Breakout / breakdown with volume confirmation
    if float(latest['Close']) > (recent_high + 0.25 * atr_last) and vol_z >= 1.0:
        return "Buy", "Breakout above recent high with volume confirmation"
    if float(latest['Close']) < (recent_low - 0.25 * atr_last) and vol_z >= 1.0:
        return "Sell", "Breakdown below recent low with volume confirmation"

    sma50 = float(latest.get('SMA_50', np.nan))
    sma200 = float(latest.get('SMA_200', np.nan)) if 'SMA_200' in latest else np.nan
    close = float(latest.get('Close', np.nan))

    # Trend checks with NaN safety
    below_both = np.isfinite(close) and np.isfinite(sma50) and (close < sma50) and (np.isfinite(sma200) and sma50 < sma200)
    above_both = np.isfinite(close) and np.isfinite(sma50) and (close > sma50) and (np.isfinite(sma200) and sma50 > sma200)

    if below_both:
        return "Sell", "Price below both SMAs; trend weakening"

    if above_both:
        # precedence bug fix: choose message based on RSI
        if float(latest.get('RSI', 50.0)) >= 70.0:
            return "Hold", "Uptrend intact; manage entries on pullbacks"
        else:
            return "Hold", "Uptrend continues; not overbought"

    return "Watch", "No clear edge; wait for breakout/bounce setup"


def get_price_targets(df):
    """Calculate price targets and risk/reward ratio using ATR and recent swings."""
    latest = df.iloc[-1]
    close = float(latest['Close'])

    try:
        atr_last = float(AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True).average_true_range().iloc[-1])
    except Exception:
        atr_last = float((df['High'] - df['Low']).tail(14).mean())
    atr_last = max(atr_last, _MIN_TOL)

    highs = df['High'].tail(50).astype(float)
    lows = df['Low'].tail(50).astype(float)
    swing_high = float(highs.max())
    swing_low = float(lows.min())
    mid = 0.5 * (swing_high + swing_low)

    # Direction: above middle -> bias up; else down
    if close >= mid:
        raw_target = close + 1.5 * atr_last
        target = max(raw_target, swing_high)  # aim beyond last swing high
        stop_loss = min(close - 1.0 * atr_last, float(lows.tail(10).min()))
        stop_loss = min(stop_loss, close - 0.5 * atr_last)  # ensure not too tight
    else:
        raw_target = close - 1.5 * atr_last
        target = min(raw_target, swing_low)   # aim beyond last swing low
        stop_loss = max(close + 1.0 * atr_last, float(highs.tail(10).max()))
        stop_loss = max(stop_loss, close + 0.5 * atr_last)

    # Risk/Reward
    if target > close:
        risk = max(_MIN_TOL, close - stop_loss)
        reward = max(0.0, target - close)
    else:
        risk = max(_MIN_TOL, stop_loss - close)
        reward = max(0.0, close - target)
    risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    return {'target': float(target), 'stop_loss': float(stop_loss), 'risk_reward': float(risk_reward)}


# ----------------------- Indicator Calculation Orchestrator -----------------

def calculate_technical_indicators(df):
    """Calculate all technical indicators and enrich the DataFrame."""
    if df.empty:
        return None

    df = df.copy()

    # RSI
    rsi = RSIIndicator(close=df['Close'], window=RSI_PERIOD)
    df['RSI'] = rsi.rsi()

    # MACD
    macd = MACD(
        close=df['Close'],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL
    )
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_histogram'] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=BB_PERIOD, window_dev=BB_STD)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()

    # Moving Averages
    for period in SMA_PERIODS:
        df[f'SMA_{period}'] = SMAIndicator(close=df['Close'], window=period).sma_indicator()

    df['EMA_20'] = EMAIndicator(close=df['Close'], window=EMA_PERIOD).ema_indicator()

    # 200-day SMA (needed for trend/health)
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()

    # ADX & ATR
    adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    df['ADX'] = adx.adx()
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    df['ATR'] = atr.average_true_range()

    # Volume Analysis
    df['Volume_SMA'] = SMAIndicator(close=df['Volume'], window=VOLUME_PERIOD).sma_indicator()
    if 'Volume' in df.columns:
        vwap = VolumeWeightedAveragePrice(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        )
        df['VWAP'] = vwap.volume_weighted_average_price()

        ad = AccDistIndexIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        )
        df['ADI'] = ad.acc_dist_index()

    # Daily returns & volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=VOLATILITY_PERIOD).std()

    # Support/Resistance
    support_levels, resistance_levels = identify_support_resistance(df)
    df['Support_1'] = support_levels[0] if len(support_levels) > 0 else None
    df['Support_2'] = support_levels[1] if len(support_levels) > 1 else None
    df['Resistance_1'] = resistance_levels[0] if len(resistance_levels) > 0 else None
    df['Resistance_2'] = resistance_levels[1] if len(resistance_levels) > 1 else None

    # Patterns
    patterns = analyze_price_patterns(df)
    for pattern, value in patterns.items():
        df[f'Pattern_{pattern}'] = value

    # Signals
    signals = generate_trading_signals(df)
    df = pd.concat([df, signals], axis=1)

    # Backtest (if enough data)
    if len(df) > 50:
        portfolio, performance = backtest_strategy(df, signals)
        for metric, value in performance.items():
            df.loc[df.index[-1], f'Backtest_{metric}'] = value

    # Health & Trend
    df['Health_Score'] = calculate_health_score(df)
    trend, strength = determine_trend(df)
    df['Trend'] = trend
    df['Trend_Strength'] = strength

    # Action & Targets
    action, reason = get_action_and_reason(df)
    df['Action'] = action
    df['Action_Reason'] = reason

    targets = get_price_targets(df)
    df['Price_Target'] = targets['target']
    df['Stop_Loss'] = targets['stop_loss']
    df['Risk_Reward'] = targets['risk_reward']

    return df
