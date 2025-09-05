"""
Data fetching module for stock data
"""

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from config import NIFTY50_SYMBOLS
from indicators import calculate_technical_indicators

def fetch_stock_data(symbol):
    """Fetch data for a single stock symbol"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get historical data for technical analysis
        hist = ticker.history(period="1y")  # Get 1 year of data
        if hist.empty:
            print(f"No historical data available for {symbol}")
            return symbol, None
            
        data = {
            'info': ticker.info,  # Get general company info
            'history': hist,  # Historical price data
            'technical': calculate_technical_indicators(hist),  # Technical indicators
        }
        return symbol, data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return symbol, None

def get_nifty50_data(symbols=None, max_workers=5):
    """Fetch data for all Nifty 50 stocks in parallel"""
    data = {}
    if symbols is None:
        symbols = NIFTY50_SYMBOLS
    max_workers = min(32, len(symbols))  # Use up to 32 workers
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = [
            executor.submit(fetch_stock_data, symbol)
            for symbol in symbols
        ]
        
        # Process the results as they complete with a progress bar
        for future in tqdm(
            futures,
            total=len(futures),
            desc="Fetching stock data"
        ):
            symbol, result = future.result()
            if result is not None:
                data[symbol] = result
    
    end_time = time.time()
    print(f"\nFetched data for {len(data)} stocks in {end_time - start_time:.2f} seconds")
    return data
