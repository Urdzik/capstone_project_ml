import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
import warnings
import requests
import time
from pathlib import Path

warnings.filterwarnings('ignore')

class DataLoader:
    """
    Клас для завантаження та обробки фінансових даних
    """
    
    def __init__(self, cache_dir: str = "../data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_stock_data(self, ticker: str, start: str, end: str, 
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Завантажує історичні дані по акції з Yahoo Finance
        """
        cache_file = self.cache_dir / f"{ticker}_{start}_{end}.csv"
        
        # Перевіряємо кеш
        if use_cache and cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Переконуємося, що індекс є datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                print(f"Завантажено з кешу: {ticker}")
                return df
            except Exception as e:
                print(f"Помилка завантаження з кешу: {e}")
        
        # Завантажуємо з Yahoo Finance
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"Немає даних для {ticker} в діапазоні {start} - {end}")
                return None
            
            # Переконуємося, що індекс є datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Очищаємо дані
            df = self.clean_data(df)
            
            # Зберігаємо в кеш
            if use_cache:
                df.to_csv(cache_file)
                
            print(f"Завантажено: {ticker} ({len(df)} записів)")
            return df
            
        except Exception as e:
            print(f"Помилка завантаження {ticker}: {e}")
            return None
    
    def load_multiple_stocks(self, tickers: List[str], start: str, end: str,
                           use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Завантажує дані для декількох акцій
        """
        data = {}
        
        for ticker in tickers:
            print(f"Завантаження {ticker}...")
            df = self.load_stock_data(ticker, start, end, use_cache)
            if df is not None:
                data[ticker] = df
            time.sleep(1)  # Щоб не перевантажувати API
        
        print(f"Завантажено {len(data)} з {len(tickers)} акцій")
        return data
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очищає та обробляє дані
        """
        # Видаляємо пропущені значення
        df = df.dropna()
        
        # Перевіряємо на аномальні значення
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                # Видаляємо значення, що дорівнюють 0
                df = df[df[col] > 0]
                
                # Видаляємо екстремальні викиди (більше 3 стандартних відхилень)
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
        
        # Перевіряємо логічність цін (High >= Low, Close між High та Low)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            df = df[df['High'] >= df['Low']]
            df = df[(df['Close'] >= df['Low']) & (df['Close'] <= df['High'])]
            df = df[(df['Open'] >= df['Low']) & (df['Open'] <= df['High'])]
        
        return df
    
    def add_stock_info(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає додаткову інформацію про акцію
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Додаємо деякі корисні метрики
            df['ticker'] = ticker
            df['sector'] = info.get('sector', 'Unknown')
            df['industry'] = info.get('industry', 'Unknown')
            df['market_cap'] = info.get('marketCap', 0)
            
        except Exception as e:
            print(f"Не вдалося отримати інформацію для {ticker}: {e}")
        
        return df
    
    def get_market_data(self, start: str, end: str) -> Dict[str, pd.DataFrame]:
        """
        Завантажує дані про ринкові індекси
        """
        market_tickers = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'DIA': 'Dow Jones',
            'VTI': 'Total Stock Market',
            'IWM': 'Russell 2000'
        }
        
        market_data = {}
        
        for ticker, name in market_tickers.items():
            df = self.load_stock_data(ticker, start, end)
            if df is not None:
                market_data[name] = df
                
        return market_data
    
    def load_crypto_data(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Завантажує дані про криптовалюти
        """
        try:
            # Для криптовалют використовуємо Yahoo Finance з суфіксом -USD
            crypto_ticker = f"{symbol}-USD"
            df = yf.download(crypto_ticker, start=start, end=end, progress=False)
            
            if df.empty:
                print(f"Немає даних для {symbol}")
                return None
            
            # Переконуємося, що індекс є datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            df = self.clean_data(df)
            print(f"Завантажено криптовалюту: {symbol}")
            return df
            
        except Exception as e:
            print(f"Помилка завантаження {symbol}: {e}")
            return None
    
    def load_forex_data(self, pair: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Завантажує дані валютних пар
        """
        try:
            # Для валютних пар використовуємо Yahoo Finance
            forex_ticker = f"{pair}=X"
            df = yf.download(forex_ticker, start=start, end=end, progress=False)
            
            if df.empty:
                print(f"Немає даних для {pair}")
                return None
            
            # Переконуємося, що індекс є datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            df = self.clean_data(df)
            print(f"Завантажено валютну пару: {pair}")
            return df
            
        except Exception as e:
            print(f"Помилка завантаження {pair}: {e}")
            return None
    
    def get_fundamental_data(self, ticker: str) -> Dict:
        """
        Отримує фундаментальні дані про компанію
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamental_data = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
            return fundamental_data
            
        except Exception as e:
            print(f"Помилка отримання фундаментальних даних для {ticker}: {e}")
            return {}
    
    def resample_data(self, df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
        """
        Перетворює дані на іншу частоту (тижнева, місячна)
        """
        try:
            resampled = df.resample(freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            print(f"Помилка ресемплювання: {e}")
            return df
    
    def split_data_by_date(self, df: pd.DataFrame, split_date: str) -> tuple:
        """
        Розділяє дані на періоди до та після вказаної дати
        """
        split_date = pd.to_datetime(split_date)
        
        before = df[df.index < split_date]
        after = df[df.index >= split_date]
        
        return before, after
    
    def get_recent_data(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Завантажує найновіші дані за вказану кількість днів
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = self.load_stock_data(ticker, start_date, end_date, use_cache=False)
        
        # Додаткова перевірка datetime індексу
        if df is not None and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        return df

# Глобальний екземпляр для зручності
data_loader = DataLoader()

# Функції для зворотної сумісності
def load_stock_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Завантажує історичні дані по акції з Yahoo Finance
    (функція для зворотної сумісності)
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            print(f"Немає даних для {ticker} в діапазоні {start} - {end}")
            return None
        
        # Переконуємося, що індекс є datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Flatten multi-level columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        print(f"Завантажено: {ticker} ({len(df)} записів)")
        return df
        
    except Exception as e:
        print(f"Помилка завантаження {ticker}: {e}")
        return None

def get_sp500_tickers() -> List[str]:
    """
    Отримує список тікерів S&P 500
    """
    try:
        # Завантажуємо список з Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Очищаємо тікери (видаляємо крапки)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        return tickers
        
    except Exception as e:
        print(f"Помилка завантаження списку S&P 500: {e}")
        # Повертаємо базовий список популярних акцій
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

def get_popular_crypto_tickers() -> List[str]:
    """
    Повертає список популярних криптовалют
    """
    return ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'SOL', 'DOT', 'AVAX']

def get_major_forex_pairs() -> List[str]:
    """
    Повертає список основних валютних пар
    """
    return ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
