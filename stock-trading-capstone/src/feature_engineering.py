import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Клас для створення технічних індикаторів та ознак
    """
    
    def __init__(self):
        self.feature_names = []
    
    def add_moving_averages(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Додає ковзні середні (SMA та EMA)
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]
        
        df = df.copy()
        
        # Simple Moving Averages
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            
        # Exponential Moving Averages
        for window in [10, 20, 50]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Moving average crossovers
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['SMA_20_50_Cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає індикатори моментуму
        """
        df = df.copy()
        
        # RSI (Relative Strength Index)
        for period in [14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Handle division by zero in RS calculation
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 100)  # If loss is 0, RSI should be 100
            rs = rs.fillna(50)  # Fill NaN with neutral RSI
            
            rsi = 100 - (100 / (1 + rs))
            df[f'RSI_{period}'] = rsi.fillna(50)  # Fill any remaining NaN with neutral RSI
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        hl_14_diff = high_14 - low_14
        
        # Safe Stochastic calculation
        stoch_k = 100 * ((df['Close'] - low_14) / hl_14_diff)
        stoch_k = stoch_k.replace([np.inf, -np.inf], 50)  # Replace inf with neutral
        stoch_k = stoch_k.fillna(50)  # Fill NaN with neutral
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_k.rolling(window=3).mean()
        
        # Williams %R (safe calculation)
        williams_r = -100 * ((high_14 - df['Close']) / hl_14_diff)
        williams_r = williams_r.replace([np.inf, -np.inf], -50)  # Replace inf with neutral
        df['Williams_R'] = williams_r.fillna(-50)
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає індикатори тренду
        """
        df = df.copy()
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # ADX (Average Directional Index) - Simplified version  
        # Calculate price momentum and trend strength
        price_change = df['Close'].pct_change()
        trend_strength = abs(price_change).rolling(14).mean() * 100
        
        # Simple directional indicators
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        
        up_move = up_move.where(up_move > 0, 0)
        down_move = down_move.where(down_move > 0, 0)
        
        df['ADX'] = trend_strength
        df['Plus_DI'] = up_move.rolling(14).mean() * 100
        df['Minus_DI'] = down_move.rolling(14).mean() * 100
        
        # Parabolic SAR (спрощена версія)
        df['PSAR'] = df['Close'].shift(1)  # Спрощена версія
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає індикатори волатильності
        """
        df = df.copy()
        
        # Bollinger Bands
        for period in [10, 20, 50]:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = sma + (std * 2)
            df[f'BB_Lower_{period}'] = sma - (std * 2)
            df[f'BB_Middle_{period}'] = sma
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
            
            # BB Position - simple and safe calculation
            bb_position = []
            for i in range(len(df)):
                try:
                    width = float(df[f'BB_Width_{period}'].iloc[i])
                    close = float(df['Close'].iloc[i])
                    lower = float(df[f'BB_Lower_{period}'].iloc[i])
                    
                    if pd.isna(width) or pd.isna(close) or pd.isna(lower) or width == 0:
                        bb_position.append(0.5)  # Neutral position
                    else:
                        pos = (close - lower) / width
                        # Clamp between 0 and 1
                        pos = max(0, min(1, pos))
                        bb_position.append(pos)
                except (ValueError, TypeError, IndexError):
                    bb_position.append(0.5)  # Default to neutral if any error
            
            df[f'BB_Position_{period}'] = bb_position
        
        # Average True Range (ATR) - manual calculation to avoid array shape issues
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
                true_range_values.append(0)  # Default value if any error
        
        true_range = pd.Series(true_range_values, index=df.index)
        
        for period in [14, 21]:
            df[f'ATR_{period}'] = true_range.rolling(period).mean()
        
        # Historical Volatility
        for period in [10, 20, 30]:
            df[f'HV_{period}'] = df['Close'].pct_change().rolling(window=period).std() * np.sqrt(252)
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає індикатори об'єму
        """
        if 'Volume' not in df.columns:
            return df
            
        df = df.copy()
        
        # Volume Moving Averages
        for period in [10, 20, 50]:
            df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
            
            # Manual volume ratio calculation to avoid DataFrame issues
            volume_ratio = []
            for i in range(len(df)):
                try:
                    volume = float(df['Volume'].iloc[i])
                    volume_sma = float(df[f'Volume_SMA_{period}'].iloc[i])
                    
                    if pd.isna(volume) or pd.isna(volume_sma) or volume_sma == 0:
                        volume_ratio.append(1.0)  # Neutral ratio
                    else:
                        ratio = volume / volume_sma
                        volume_ratio.append(ratio)
                except (ValueError, TypeError, IndexError):
                    volume_ratio.append(1.0)  # Default to neutral if any error
            
            df[f'Volume_Ratio_{period}'] = volume_ratio
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume Price Trend (VPT)
        df['VPT'] = df['Volume'] * (df['Close'].pct_change())
        df['VPT'] = df['VPT'].fillna(0).cumsum()
        
        # Accumulation/Distribution Line
        hl_diff = df['High'] - df['Low']
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_diff
        clv = clv.replace([np.inf, -np.inf], 0)  # Replace inf values
        clv = clv.fillna(0)
        df['AD_Line'] = (clv * df['Volume']).cumsum()
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        
        # Safe MFI calculation
        money_ratio = positive_flow / negative_flow
        money_ratio = money_ratio.replace([np.inf, -np.inf], 100)  # If negative_flow is 0, MFI should be 100
        money_ratio = money_ratio.fillna(50)  # Fill NaN with neutral ratio
        
        mfi = 100 - (100 / (1 + money_ratio))
        df['MFI'] = mfi.fillna(50)  # Fill any remaining NaN with neutral MFI
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає ознаки на основі цін
        """
        df = df.copy()
        
        # Price changes
        for period in [1, 2, 3, 5, 10]:
            df[f'Price_Change_{period}'] = df['Close'].pct_change(periods=period)
            df[f'Price_Change_Abs_{period}'] = df[f'Price_Change_{period}'].abs()
        
        # High-Low spread
        df['HL_Spread'] = df['High'] - df['Low']
        
        # Manual HL Spread Pct calculation
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
        
        # Open-Close spread  
        df['OC_Spread'] = df['Close'] - df['Open']
        
        # Manual OC Spread Pct calculation
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
        
        # Position within daily range - manual calculation
        position_in_range = []
        for i in range(len(df)):
            try:
                high = float(df['High'].iloc[i])
                low = float(df['Low'].iloc[i])
                close = float(df['Close'].iloc[i])
                
                hl_range = high - low
                if pd.isna(high) or pd.isna(low) or pd.isna(close) or hl_range == 0:
                    position_in_range.append(0.5)  # Neutral position
                else:
                    pos = (close - low) / hl_range
                    position_in_range.append(pos)
            except (ValueError, TypeError, IndexError):
                position_in_range.append(0.5)
        df['Position_in_Range'] = position_in_range
        
        # Gap indicators
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        
        # Manual Gap Pct calculation
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
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            for col in ['Close', 'Volume', 'High', 'Low']:
                if col in df.columns:
                    df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Додає статистичні ознаки
        """
        if windows is None:
            windows = [5, 10, 20, 50]
            
        df = df.copy()
        
        for window in windows:
            # Rolling statistics
            df[f'Close_Mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_Std_{window}'] = df['Close'].rolling(window).std()
            df[f'Close_Min_{window}'] = df['Close'].rolling(window).min()
            df[f'Close_Max_{window}'] = df['Close'].rolling(window).max()
            df[f'Close_Median_{window}'] = df['Close'].rolling(window).median()
            
            # Z-score (manual calculation)
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
            
            # Percentile rank
            df[f'Percentile_Rank_{window}'] = df['Close'].rolling(window).rank(pct=True)
            
        return df
    
    def add_fourier_features(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """
        Додає ознаки на основі перетворення Фур'є
        """
        df = df.copy()
        
        # Apply FFT to close prices
        close_fft = np.fft.fft(df['Close'].ffill())
        fft_df = pd.DataFrame({'fft': close_fft})
        
        # Extract frequency components
        for i in range(n_components):
            df[f'FFT_Real_{i}'] = np.real(close_fft)[i:len(df)+i][:len(df)]
            df[f'FFT_Imag_{i}'] = np.imag(close_fft)[i:len(df)+i][:len(df)]
        
        return df
    
    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додає циклічні ознаки на основі дати
        """
        df = df.copy()
        df = df.reset_index()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif df.index.name == 'Date' or hasattr(df.index, 'date'):
            df['Date'] = df.index
        else:
            return df.set_index('Date') if 'Date' in df.columns else df
        
        # Day of week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Month
        df['Month'] = df['Date'].dt.month
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Quarter
        df['Quarter'] = df['Date'].dt.quarter
        df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        
        return df.set_index('Date')
    
    def add_all_features(self, df: pd.DataFrame, include_fourier: bool = False) -> pd.DataFrame:
        """
        Додає всі технічні індикатори та ознаки
        """
        print("Додавання ковзних середніх...")
        df = self.add_moving_averages(df)
        
        print("Додавання індикаторів моментуму...")
        df = self.add_momentum_indicators(df)
        
        print("Додавання індикаторів тренду...")
        df = self.add_trend_indicators(df)
        
        print("Додавання індикаторів волатильності...")
        df = self.add_volatility_indicators(df)
        
        print("Додавання індикаторів об'єму...")
        df = self.add_volume_indicators(df)
        
        print("Додавання цінових ознак...")
        df = self.add_price_features(df)
        
        print("Додавання статистичних ознак...")
        df = self.add_statistical_features(df)
        
        print("Додавання циклічних ознак...")
        df = self.add_cyclical_features(df)
        
        if include_fourier:
            print("Додавання ознак Фур'є...")
            df = self.add_fourier_features(df)
        
        print(f"Загалом створено {len(df.columns)} ознак")
        return df

# Глобальний екземпляр для зручності
feature_engineer = FeatureEngineer()

# Функції для зворотної сумісності
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає набір класичних технічних індикаторів (функція для зворотної сумісності)
    """
    return feature_engineer.add_all_features(df, include_fourier=False)

def create_features(df: pd.DataFrame, feature_types: List[str] = None) -> pd.DataFrame:
    """
    Створює ознаки відповідно до вказаних типів
    """
    if feature_types is None:
        feature_types = ['moving_averages', 'momentum', 'trend', 'volatility', 'volume', 'price', 'statistical']
    
    engineer = FeatureEngineer()
    result_df = df.copy()
    
    for feature_type in feature_types:
        if feature_type == 'moving_averages':
            result_df = engineer.add_moving_averages(result_df)
        elif feature_type == 'momentum':
            result_df = engineer.add_momentum_indicators(result_df)
        elif feature_type == 'trend':
            result_df = engineer.add_trend_indicators(result_df)
        elif feature_type == 'volatility':
            result_df = engineer.add_volatility_indicators(result_df)
        elif feature_type == 'volume':
            result_df = engineer.add_volume_indicators(result_df)
        elif feature_type == 'price':
            result_df = engineer.add_price_features(result_df)
        elif feature_type == 'statistical':
            result_df = engineer.add_statistical_features(result_df)
        elif feature_type == 'cyclical':
            result_df = engineer.add_cyclical_features(result_df)
        elif feature_type == 'fourier':
            result_df = engineer.add_fourier_features(result_df)
    
    return result_df

def get_feature_importance_analysis(df: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
    """
    Аналізує важливість ознак для прогнозування
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    
    # Підготовка даних
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # Mutual information
    mi_scores = mutual_info_regression(X, y)
    
    # Створення DataFrame з результатами
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf_importance,
        'mutual_info': mi_scores
    })
    
    # Комбінована важливість
    importance_df['combined_importance'] = (
        importance_df['rf_importance'] + importance_df['mutual_info']
    ) / 2
    
    return importance_df.sort_values('combined_importance', ascending=False) 