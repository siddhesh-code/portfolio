"""
Data fetching module for stock data
"""

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        futures = {
            executor.submit(fetch_stock_data, symbol): symbol
            for symbol in symbols
        }

        with tqdm(
            total=len(futures),
            desc="Fetching stock data"
        ) as progress:
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    symbol, result = future.result()
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"Error fetching data for {symbol}: {exc}")
                    progress.update(1)
                    continue

                if result is not None:
                    data[symbol] = result

                progress.update(1)
    
    end_time = time.time()
    print(f"\nFetched data for {len(data)} stocks in {end_time - start_time:.2f} seconds")
    return data
