import pandas as pd
import numpy as np
from typing import List, Optional, Dict

class FeatureEngineer:
    
    def __init__(self):
        self.feature_names = []
    
    def add_moving_averages(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]
        
        df = df.copy()
        
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            
        for window in [10, 20, 50]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        

        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['SMA_20_50_Cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        

        for period in [14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            

            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 100)
            rs = rs.fillna(50)
            
            rsi = 100 - (100 / (1 + rs))
            df[f'RSI_{period}'] = rsi.fillna(50)
        

        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        hl_14_diff = high_14 - low_14
        

        stoch_k = 100 * ((df['Close'] - low_14) / hl_14_diff)
        stoch_k = stoch_k.replace([np.inf, -np.inf], 50)
        stoch_k = stoch_k.fillna(50)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_k.rolling(window=3).mean()
        

        williams_r = -100 * ((high_14 - df['Close']) / hl_14_diff)
        williams_r = williams_r.replace([np.inf, -np.inf], -50)
        df['Williams_R'] = williams_r.fillna(-50)
        

        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100
        

        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        price_change = df['Close'].pct_change()
        trend_strength = abs(price_change).rolling(14).mean() * 100
        
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        
        up_move = up_move.where(up_move > 0, 0)
        down_move = down_move.where(down_move > 0, 0)
        
        df['ADX'] = trend_strength
        df['Plus_DI'] = up_move.rolling(14).mean() * 100
        df['Minus_DI'] = down_move.rolling(14).mean() * 100
        
        df['PSAR'] = df['Close'].shift(1)
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        for period in [10, 20, 50]:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = sma + (std * 2)
            df[f'BB_Lower_{period}'] = sma - (std * 2)
            df[f'BB_Middle_{period}'] = sma
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
            
            bb_position = []
            for i in range(len(df)):
                try:
                    width = float(df[f'BB_Width_{period}'].iloc[i])
                    close = float(df['Close'].iloc[i])
                    lower = float(df[f'BB_Lower_{period}'].iloc[i])
                    
                    if pd.isna(width) or pd.isna(close) or pd.isna(lower) or width == 0:
                        bb_position.append(0.5)
                    else:
                        pos = (close - lower) / width
                        pos = max(0, min(1, pos))
                        bb_position.append(pos)
                except (ValueError, TypeError, IndexError):
                    bb_position.append(0.5)
            
            df[f'BB_Position_{period}'] = bb_position
        
        true_range_values = []
        for i in range(len(df)):
            try:
                high = float(df['High'].iloc[i])
                low = float(df['Low'].iloc[i])
                prev_close = float(df['Close'].iloc[i-1]) if i > 0 else float(df['Close'].iloc[i])
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range_values.append(max(tr1, tr2, tr3))
            except (ValueError, TypeError, IndexError):
                true_range_values.append(0)
        
        true_range = pd.Series(true_range_values, index=df.index)
        
        for period in [14, 21]:
            df[f'ATR_{period}'] = true_range.rolling(period).mean()
        
        for period in [10, 20, 30]:
            df[f'HV_{period}'] = df['Close'].pct_change().rolling(window=period).std() * np.sqrt(252)
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if 'Volume' not in df.columns:
            return df
            
        df = df.copy()
        
        for period in [10, 20, 50]:
            df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
            
            volume_ratio = []
            for i in range(len(df)):
                try:
                    volume = float(df['Volume'].iloc[i])
                    volume_sma = float(df[f'Volume_SMA_{period}'].iloc[i])
                    
                    if pd.isna(volume) or pd.isna(volume_sma) or volume_sma == 0:
                        volume_ratio.append(1.0)
                    else:
                        ratio = volume / volume_sma
                        volume_ratio.append(ratio)
                except (ValueError, TypeError, IndexError):
                    volume_ratio.append(1.0)
            
            df[f'Volume_Ratio_{period}'] = volume_ratio
        
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        df['VPT'] = df['Volume'] * (df['Close'].pct_change())
        df['VPT'] = df['VPT'].fillna(0).cumsum()
        
        hl_diff = df['High'] - df['Low']
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_diff
        clv = clv.replace([np.inf, -np.inf], 0)
        clv = clv.fillna(0)
        df['AD_Line'] = (clv * df['Volume']).cumsum()
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        
        money_ratio = positive_flow / negative_flow
        money_ratio = money_ratio.replace([np.inf, -np.inf], 100)
        money_ratio = money_ratio.fillna(50)
        
        mfi = 100 - (100 / (1 + money_ratio))
        df['MFI'] = mfi.fillna(50)
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        for period in [1, 2, 3, 5, 10]:
            df[f'Price_Change_{period}'] = df['Close'].pct_change(periods=period)
            df[f'Price_Change_Abs_{period}'] = df[f'Price_Change_{period}'].abs()
        
        df['HL_Spread'] = df['High'] - df['Low']
        
        hl_spread_pct = []
        for i in range(len(df)):
            try:
                spread = float(df['HL_Spread'].iloc[i])
                close = float(df['Close'].iloc[i])
                if pd.isna(spread) or pd.isna(close) or close == 0:
                    hl_spread_pct.append(0)
                else:
                    hl_spread_pct.append(spread / close)
            except (ValueError, TypeError, IndexError):
                hl_spread_pct.append(0)
        df['HL_Spread_Pct'] = hl_spread_pct
        
        df['OC_Spread'] = df['Close'] - df['Open']
        
        oc_spread_pct = []
        for i in range(len(df)):
            try:
                spread = float(df['OC_Spread'].iloc[i])
                open_price = float(df['Open'].iloc[i])
                if pd.isna(spread) or pd.isna(open_price) or open_price == 0:
                    oc_spread_pct.append(0)
                else:
                    oc_spread_pct.append(spread / open_price)
            except (ValueError, TypeError, IndexError):
                oc_spread_pct.append(0)
        df['OC_Spread_Pct'] = oc_spread_pct
        
        position_in_range = []
        for i in range(len(df)):
            try:
                high = float(df['High'].iloc[i])
                low = float(df['Low'].iloc[i])
                close = float(df['Close'].iloc[i])
                
                hl_range = high - low
                if pd.isna(high) or pd.isna(low) or pd.isna(close) or hl_range == 0:
                    position_in_range.append(0.5)
                else:
                    pos = (close - low) / hl_range
                    position_in_range.append(pos)
            except (ValueError, TypeError, IndexError):
                position_in_range.append(0.5)
        df['Position_in_Range'] = position_in_range
        
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        
        gap_pct = []
        for i in range(len(df)):
            try:
                gap = float(df['Gap'].iloc[i])
                prev_close = float(df['Close'].iloc[i-1]) if i > 0 else float(df['Close'].iloc[i])
                
                if pd.isna(gap) or pd.isna(prev_close) or prev_close == 0:
                    gap_pct.append(0)
                else:
                    gap_pct.append(gap / prev_close)
            except (ValueError, TypeError, IndexError):
                gap_pct.append(0)
        df['Gap_Pct'] = gap_pct
        
        for lag in [1, 2, 3, 5, 10]:
            for col in ['Close', 'Volume', 'High', 'Low']:
                if col in df.columns:
                    df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        
        if windows is None:
            windows = [5, 10, 20, 50]
            
        df = df.copy()
        
        for window in windows:
            df[f'Close_Mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_Std_{window}'] = df['Close'].rolling(window).std()
            df[f'Close_Min_{window}'] = df['Close'].rolling(window).min()
            df[f'Close_Max_{window}'] = df['Close'].rolling(window).max()
            df[f'Close_Median_{window}'] = df['Close'].rolling(window).median()
            
            z_score = []
            for i in range(len(df)):
                try:
                    close = float(df['Close'].iloc[i])
                    mean = float(df[f'Close_Mean_{window}'].iloc[i])
                    std = float(df[f'Close_Std_{window}'].iloc[i])
                    
                    if pd.isna(close) or pd.isna(mean) or pd.isna(std) or std == 0:
                        z_score.append(0)
                    else:
                        z_score.append((close - mean) / std)
                except (ValueError, TypeError, IndexError):
                    z_score.append(0)
            df[f'Z_Score_{window}'] = z_score
            
            df[f'Percentile_Rank_{window}'] = df['Close'].rolling(window).rank(pct=True)
            
        return df
    
    def add_fourier_features(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        
        df = df.copy()
        
        close_fft = np.fft.fft(df['Close'].ffill())
        fft_df = pd.DataFrame({'fft': close_fft})
        
        for i in range(n_components):
            df[f'FFT_Real_{i}'] = np.real(close_fft)[i:len(df)+i][:len(df)]
            df[f'FFT_Imag_{i}'] = np.imag(close_fft)[i:len(df)+i][:len(df)]
        
        return df
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        df = df.reset_index()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif df.index.name == 'Date' or hasattr(df.index, 'date'):
            df['Date'] = df.index
        else:
            return df.set_index('Date') if 'Date' in df.columns else df
        
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        df['Month'] = df['Date'].dt.month
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        df['Quarter'] = df['Date'].dt.quarter
        df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        
        return df.set_index('Date')
    
    def add_all_features(self, df: pd.DataFrame, include_fourier: bool = False) -> pd.DataFrame:
        
        df = self.add_moving_averages(df)
        
        df = self.add_momentum_indicators(df)
        
        df = self.add_trend_indicators(df)
        
        df = self.add_volatility_indicators(df)

        df = self.add_volume_indicators(df)
        
        df = self.add_price_features(df)
        
        df = self.add_statistical_features(df)
        
        df = self.add_cyclical_features(df)
        
        if include_fourier:
            df = self.add_fourier_features(df)
        
        return df

feature_engineer = FeatureEngineer()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    return feature_engineer.add_all_features(df, include_fourier=False)

