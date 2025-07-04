import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

class DataLoader:
    
    def __init__(self, cache_dir: str = "../data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_stock_data(self, ticker: str, start: str, end: str, 
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
        cache_file = self.cache_dir / f"{ticker}_{start}_{end}.csv"
        
        if use_cache and cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                print(f"Loaded from cache: {ticker}")
                return df
            except Exception as e:
                print(f"Cache load error: {e}")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"No data for {ticker} in range {start} - {end}")
                return None
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            df = self.clean_data(df)
            
            if use_cache:
                df.to_csv(cache_file)
                
            print(f"Loaded: {ticker} ({len(df)} records)")
            return df
            
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            return None
    

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df = df[df[col] > 0]
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            df = df[df['High'] >= df['Low']]
            df = df[(df['Close'] >= df['Low']) & (df['Close'] <= df['High'])]
            df = df[(df['Open'] >= df['Low']) & (df['Open'] <= df['High'])]
        
        return df



data_loader = DataLoader()

def load_stock_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            print(f"No data for {ticker} in range {start} - {end}")
            return None
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        print(f"Loaded: {ticker} ({len(df)} records)")
        return df
        
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None

