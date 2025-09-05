"""
Main entry point for the portfolio analysis application
"""

from data_fetcher import get_nifty50_data
from stock_analyzer import format_stock_analysis

def analyze_stocks(symbols, max_workers=5):
    """
    Analyze multiple stocks from Nifty 50 in parallel
    """
    # Fetch stock data
    results = get_nifty50_data(symbols, max_workers)
    
    # Format and return the analysis
    return format_stock_analysis(results)

def main():
    """
    Main entry point of the application
    """
    # Example usage with a few Nifty 50 symbols
    symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS"
    ]
    
    # Get analysis results
    analysis = analyze_stocks(symbols)
    
    # Print formatted results
    for stock in analysis:
        print(f"\nAnalysis for {stock['symbol']}:")
        print(f"Price: {stock['price']:.2f}")
        print(f"Volume: {stock['volume']}")
        print(f"Health Score: {stock['health_score']}")
        print(f"Trend: {stock['trend']} ({stock['trend_strength']})")
        print(f"Action: {stock['action']} - {stock['action_reason']}")
        print(f"Price Change: {stock['price_change']:.2f} ({stock['day_change']:.2f}%)")
        print(f"Price Target: {stock['price_target']:.2f}")
        print(f"Stop Loss: {stock['stop_loss']:.2f}")
        print(f"Risk/Reward: {stock['risk_reward']:.2f}")
        
        print("\nIndicators:")
        ind = stock['indicators']
        
        # RSI
        rsi = ind['RSI']
        print(f"\nRSI Analysis:")
        print(f"  Value: {rsi['value']:.2f}")
        print(f"  Signal: {rsi['signal']}")
        
        # MACD
        macd = ind['MACD']
        print(f"\nMACD Analysis:")
        print(f"  MACD: {macd['MACD']:.2f}")
        print(f"  Signal: {macd['Signal']:.2f}")
        print(f"  Histogram: {macd['Histogram']:.2f}")
        print(f"  Trend: {macd['Trend']}")
        
        # Support/Resistance
        sr = ind['Support_Resistance']
        print(f"\nSupport/Resistance Levels:")
        print(f"  Resistance 2: {sr['Resistance_2']:.2f}")
        print(f"  Resistance 1: {sr['Resistance_1']:.2f}")
        print(f"  Support 1: {sr['Support_1']:.2f}")
        print(f"  Support 2: {sr['Support_2']:.2f}")
        
        # Bollinger Bands
        bb = ind['BB']
        print(f"\nBollinger Bands:")
        print(f"  Upper: {bb['Upper']:.2f}")
        print(f"  Middle: {bb['Middle']:.2f}")
        print(f"  Lower: {bb['Lower']:.2f}")
        position, signal = bb['Position']
        print(f"  Position: {position:.2f}% - {signal}")
        
        # Moving Averages
        sma = ind['SMA']
        print(f"\nMoving Averages:")
        print(f"  SMA 20: {sma['SMA_20']:.2f}")
        print(f"  SMA 50: {sma['SMA_50']:.2f}")
        print(f"  Trend: {sma['Trend']}")
        
        # Volume
        vol = ind['Volume']
        print(f"\nVolume Analysis:")
        print(f"  Current: {vol['Current']:,}")
        print(f"  20-day Average: {vol['SMA_20']:,.0f}")
        print(f"  Trend: {vol['Trend']}")
        
        print(f"\nVolatility: {ind['Volatility']:.2f}%")
        
        # Backtest Performance
        if 'backtest' in stock:
            bt = stock['backtest']
            print("\nBacktest Performance:")
            print(f"  Total Return: {bt['total_return']*100:.2f}%")
            print(f"  Annual Return: {bt['annual_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {bt['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {bt['max_drawdown']*100:.2f}%")
            
        print("-" * 50)

if __name__ == "__main__":
    main()