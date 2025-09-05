"""
Technical indicators calculation module with time series analysis and trading signals
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice, AccDistIndexIndicator
from scipy.signal import argrelextrema
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, VOLUME_PERIOD, VOLATILITY_PERIOD,
    SMA_PERIODS, EMA_PERIOD
)

def identify_support_resistance(df, window=20, num_points=5):
    """
    Identify support and resistance levels using local minima and maxima
    """
    # Find local minima and maxima
    local_max = argrelextrema(df['High'].values, np.greater, order=window)[0]
    local_min = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    resistance_levels = sorted(df['High'].iloc[local_max].unique(), reverse=True)[:num_points]
    support_levels = sorted(df['Low'].iloc[local_min].unique())[:num_points]
    
    return support_levels, resistance_levels

def analyze_price_patterns(df, window=20):
    """
    Identify common price patterns
    Returns a dictionary with pattern signals
    """
    patterns = {
        'double_top': False,
        'double_bottom': False,
        'head_shoulders': False,
        'trend': 'neutral'
    }
    
    # Identify trend
    if df['SMA_50'].iloc[-1] > df['SMA_50'].iloc[-window]:
        patterns['trend'] = 'uptrend'
    elif df['SMA_50'].iloc[-1] < df['SMA_50'].iloc[-window]:
        patterns['trend'] = 'downtrend'
    
    # Double top/bottom detection (simplified)
    highs = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    if len(highs) >= 2:
        last_two_highs = df['High'].iloc[highs[-2:]].values
        if np.abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] < 0.02:
            patterns['double_top'] = True
    
    if len(lows) >= 2:
        last_two_lows = df['Low'].iloc[lows[-2:]].values
        if np.abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] < 0.02:
            patterns['double_bottom'] = True
    
    return patterns

def generate_trading_signals(df):
    """
    Generate trading signals based on multiple indicators
    Returns DataFrame with signals: 1 (buy), -1 (sell), 0 (hold)
    """
    signals = pd.DataFrame(index=df.index)
    
    # RSI signals
    signals['rsi_signal'] = 0
    signals.loc[df['RSI'] < 30, 'rsi_signal'] = 1
    signals.loc[df['RSI'] > 70, 'rsi_signal'] = -1
    
    # MACD signals
    signals['macd_signal'] = 0
    signals.loc[df['MACD'] > df['MACD_signal'], 'macd_signal'] = 1
    signals.loc[df['MACD'] < df['MACD_signal'], 'macd_signal'] = -1
    
    # Bollinger Bands signals
    signals['bb_signal'] = 0
    signals.loc[df['Close'] < df['BB_lower'], 'bb_signal'] = 1
    signals.loc[df['Close'] > df['BB_upper'], 'bb_signal'] = -1
    
    # Moving Average signals
    signals['ma_signal'] = 0
    signals.loc[df['SMA_20'] > df['SMA_50'], 'ma_signal'] = 1
    signals.loc[df['SMA_20'] < df['SMA_50'], 'ma_signal'] = -1
    
    # Volume signals
    signals['volume_signal'] = 0
    signals.loc[df['Volume'] > df['Volume_SMA'], 'volume_signal'] = 1
    
    # Combined signal (weighted average)
    weights = {
        'rsi_signal': 0.2,
        'macd_signal': 0.3,
        'bb_signal': 0.2,
        'ma_signal': 0.2,
        'volume_signal': 0.1
    }
    
    signals['combined_signal'] = sum(signals[col] * weight for col, weight in weights.items())
    signals['position'] = 0
    signals.loc[signals['combined_signal'] > 0.3, 'position'] = 1  # Buy signal
    signals.loc[signals['combined_signal'] < -0.3, 'position'] = -1  # Sell signal
    
    return signals

def backtest_strategy(df, signals, initial_capital=100000.0):
    """
    Backtest the trading strategy
    Returns DataFrame with portfolio value and performance metrics
    """
    position = pd.DataFrame(index=signals.index)
    portfolio = pd.DataFrame(index=signals.index)
    
    # Position
    position['position'] = signals['position']
    
    # Holdings
    portfolio['holdings'] = (position['position'] * df['Close'])
    
    # Cash
    portfolio['cash'] = initial_capital - (position['position'].diff() * df['Close']).cumsum()
    
    # Total
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    
    # Returns
    portfolio['returns'] = portfolio['total'].pct_change()
    
    # Performance metrics
    performance = {
        'total_return': (portfolio['total'].iloc[-1] - initial_capital) / initial_capital,
        'annual_return': portfolio['returns'].mean() * 252,
        'sharpe_ratio': portfolio['returns'].mean() / portfolio['returns'].std() * np.sqrt(252),
        'max_drawdown': (portfolio['total'] / portfolio['total'].expanding(min_periods=1).max() - 1).min()
    }
    
    return portfolio, performance

def calculate_health_score(df):
    """Calculate stock health score (0-100)"""
    # Get latest data
    latest = df.iloc[-1]
    score = 0
    
    # Trend score (30 points)
    if latest['SMA_50'] > latest['SMA_200']:
        score += 15
        if latest['Close'] > latest['SMA_50']:
            score += 15
    
    # Volume score (20 points)
    if latest['Volume'] > latest['Volume_SMA']:
        score += 20
    
    # Momentum score (30 points)
    if latest['RSI'] > 50:
        score += 15
    if latest['MACD'] > latest['MACD_signal']:
        score += 15
    
    # Volatility score (20 points)
    bb_width = (latest['BB_upper'] - latest['BB_lower']) / latest['BB_middle']
    if bb_width < 0.1:  # Low volatility
        score += 20
    elif bb_width < 0.2:  # Moderate volatility
        score += 10
    
    return min(100, score)

def determine_trend(df):
    """Determine trend direction and strength"""
    latest = df.iloc[-1]
    sma50 = latest['SMA_50']
    sma200 = latest['SMA_200']
    close = latest['Close']
    
    # Trend direction
    if close > sma50 > sma200:
        trend = "Uptrend"
        strength = "Strong" if (close - sma200)/sma200 > 0.05 else "Moderate"
    elif close < sma50 < sma200:
        trend = "Downtrend"
        strength = "Strong" if (sma200 - close)/sma200 > 0.05 else "Moderate"
    else:
        trend = "Sideways"
        strength = "Weak"
    
    return trend, strength

def get_action_and_reason(df):
    """Determine trading action and provide reason"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Check for breakout
    recent_high = df['High'].tail(20).max()
    recent_low = df['Low'].tail(20).min()
    
    if latest['Close'] > recent_high and latest['Volume'] > latest['Volume_SMA']:
        return "Buy", "Price broke above recent high with strong volume"
    
    if latest['Close'] < recent_low and latest['Volume'] > latest['Volume_SMA']:
        return "Sell", "Price broke below recent low with high volume"
    
    if latest['Close'] < latest['SMA_50'] < latest['SMA_200']:
        return "Sell", "Price below both moving averages, trend weakening"
    
    if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
        if latest['RSI'] < 70:
            return "Hold", "Strong uptrend continues but not overbought"
        else:
            return "Hold", "Uptrend but overbought, wait for pullback"
    
    return "Watch", "No clear trend, monitor for breakout"

def get_price_targets(df):
    """Calculate price targets and risk/reward ratio"""
    latest = df.iloc[-1]
    
    # Find recent pivot points
    highs = df['High'].tail(50)
    lows = df['Low'].tail(50)
    
    # Target based on recent price action
    recent_range = highs.max() - lows.min()
    target = latest['Close'] + recent_range
    
    # Stop loss based on recent support
    stop_loss = max(
        latest['Close'] * 0.95,  # Maximum 5% loss
        lows.tail(10).min()  # Recent low
    )
    
    # Calculate risk/reward
    risk = latest['Close'] - stop_loss
    reward = target - latest['Close']
    risk_reward = round(reward / risk, 1) if risk > 0 else 0
    
    return {
        'target': target,
        'stop_loss': stop_loss,
        'risk_reward': risk_reward
    }

def calculate_technical_indicators(df):
    """Calculate all technical indicators for a given DataFrame"""
    if df.empty:
        return None
        
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
    bb = BollingerBands(
        close=df['Close'],
        window=BB_PERIOD,
        window_dev=BB_STD
    )
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    
    # Moving Averages
    for period in SMA_PERIODS:
        df[f'SMA_{period}'] = SMAIndicator(
            close=df['Close'],
            window=period
        ).sma_indicator()
    
    df['EMA_20'] = EMAIndicator(
        close=df['Close'],
        window=EMA_PERIOD
    ).ema_indicator()
    
    # Volume Analysis
    df['Volume_SMA'] = SMAIndicator(
        close=df['Volume'],
        window=VOLUME_PERIOD
    ).sma_indicator()
    
    if 'Volume' in df.columns:
        vwap = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        )
        df['VWAP'] = vwap.volume_weighted_average_price()
        
        # Accumulation/Distribution Index
        ad = AccDistIndexIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        )
        df['ADI'] = ad.acc_dist_index()
    
    # Calculate price changes and volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(
        window=VOLATILITY_PERIOD
    ).std()
    
    # Add support/resistance levels
    support_levels, resistance_levels = identify_support_resistance(df)
    df['Support_1'] = support_levels[0] if len(support_levels) > 0 else None
    df['Support_2'] = support_levels[1] if len(support_levels) > 1 else None
    df['Resistance_1'] = resistance_levels[0] if len(resistance_levels) > 0 else None
    df['Resistance_2'] = resistance_levels[1] if len(resistance_levels) > 1 else None
    
    # Add price patterns
    patterns = analyze_price_patterns(df)
    for pattern, value in patterns.items():
        df[f'Pattern_{pattern}'] = value
    
    # Generate trading signals
    signals = generate_trading_signals(df)
    df = pd.concat([df, signals], axis=1)
    
    # Run backtest if we have enough data
    if len(df) > 50:  # Minimum data requirement
        portfolio, performance = backtest_strategy(df, signals)
        
        # Add performance metrics to the last row
        for metric, value in performance.items():
            df.loc[df.index[-1], f'Backtest_{metric}'] = value
    
    # Calculate 200-day SMA for trend strength
    df['SMA_200'] = SMAIndicator(
        close=df['Close'],
        window=200
    ).sma_indicator()
    
    # Calculate health score and trend
    df['Health_Score'] = calculate_health_score(df)
    trend, strength = determine_trend(df)
    df['Trend'] = trend
    df['Trend_Strength'] = strength
    
    # Get trading action and targets
    action, reason = get_action_and_reason(df)
    df['Action'] = action
    df['Action_Reason'] = reason
    
    # Calculate price targets
    targets = get_price_targets(df)
    df['Price_Target'] = targets['target']
    df['Stop_Loss'] = targets['stop_loss']
    df['Risk_Reward'] = targets['risk_reward']
    
    return df
