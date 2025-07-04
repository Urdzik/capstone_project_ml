import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class TechnicalIndicators:
    
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(window).mean()
        
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(window).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        return (typical_price - sma) / (0.015 * mean_deviation)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()
        
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

def add_simple_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df.columns = [str(col) for col in df.columns]
    
    indicators = TechnicalIndicators()
    
    df['SMA_20'] = indicators.sma(df['Close'], 20)
    df['SMA_50'] = indicators.sma(df['Close'], 50)
    df['EMA_12'] = indicators.ema(df['Close'], 12)
    df['EMA_26'] = indicators.ema(df['Close'], 26)
    
    df['RSI'] = indicators.rsi(df['Close'])
    
    macd_line, signal_line, histogram = indicators.macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = histogram
    
    bb_upper, bb_middle, bb_lower = indicators.bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    
    df['Price_SMA_Ratio'] = df['Close'] / df['SMA_20'].fillna(1)
    
    if 'Volume' in df.columns:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].fillna(1)
        df['OBV'] = indicators.obv(df['Close'], df['Volume'])
    
    for lag in [1, 2, 3]:
        df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
    
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_Lag_1'] = df['Price_Change'].shift(1)
    
    return df

def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    indicators = TechnicalIndicators()
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame повинен містити колонки: {required_cols}")
    
    df['RSI_14'] = indicators.rsi(df['Close'], 14)
    df['RSI_21'] = indicators.rsi(df['Close'], 21)
    
    stoch_k, stoch_d = indicators.stochastic(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    
    df['Williams_R'] = indicators.williams_r(df['High'], df['Low'], df['Close'])
    
    df['ATR'] = indicators.atr(df['High'], df['Low'], df['Close'])
    
    adx, plus_di, minus_di = indicators.adx(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    
    df['CCI'] = indicators.cci(df['High'], df['Low'], df['Close'])
    
    if 'Volume' in df.columns:
        df['MFI'] = indicators.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        df['VWAP'] = indicators.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    pivot_data = indicators.pivot_points(df['High'], df['Low'], df['Close'])
    for key, value in pivot_data.items():
        df[f'Pivot_{key.upper()}'] = value
    
    ichimoku_data = indicators.ichimoku(df['High'], df['Low'], df['Close'])
    for key, value in ichimoku_data.items():
        df[f'Ichimoku_{key.title()}'] = value
    
    return df

def get_indicator_signals(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    signals = pd.DataFrame(index=df.index)
    
    if 'RSI' in df.columns:
        signals['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        signals['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        signals['MACD_Bull'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        signals['MACD_Bear'] = (df['MACD'] < df['MACD_Signal']).astype(int)
    
    if all(col in df.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
        signals['BB_Upper_Break'] = (df['Close'] > df['BB_Upper']).astype(int)
        signals['BB_Lower_Break'] = (df['Close'] < df['BB_Lower']).astype(int)
    
    if all(col in df.columns for col in ['Close', 'SMA_20', 'SMA_50']):
        signals['SMA_Golden_Cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        signals['SMA_Death_Cross'] = (df['SMA_20'] < df['SMA_50']).astype(int)
        signals['Price_Above_SMA20'] = (df['Close'] > df['SMA_20']).astype(int)
    
    if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
        signals['Stoch_Oversold'] = ((df['Stoch_K'] < 20) & (df['Stoch_D'] < 20)).astype(int)
        signals['Stoch_Overbought'] = ((df['Stoch_K'] > 80) & (df['Stoch_D'] > 80)).astype(int)
    
    bull_signals = [col for col in signals.columns if 'Bull' in col or 'Golden' in col or 'Oversold' in col]
    bear_signals = [col for col in signals.columns if 'Bear' in col or 'Death' in col or 'Overbought' in col]
    
    if bull_signals:
        signals['Combined_Bull'] = signals[bull_signals].sum(axis=1)
    if bear_signals:
        signals['Combined_Bear'] = signals[bear_signals].sum(axis=1)
    
    return signals

def calculate_indicator_statistics(df: pd.DataFrame) -> pd.DataFrame:
    
    indicator_cols = [col for col in df.columns if any(indicator in col.upper() 
                     for indicator in ['RSI', 'MACD', 'BB', 'SMA', 'EMA', 'STOCH', 'ATR', 'ADX'])]
    
    stats = []
    
    for col in indicator_cols:
        if col in df.columns:
            col_stats = {
                'indicator': col,
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'current': df[col].iloc[-1] if len(df) > 0 else np.nan,
                'percentile_25': df[col].quantile(0.25),
                'percentile_75': df[col].quantile(0.75)
            }
            stats.append(col_stats)
    
    return pd.DataFrame(stats) 