import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

class TradingStrategy(ABC):
    
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio_value = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        
        pass
    
    def calculate_position_size(self, price: float, signal: int) -> float:
        
        if signal == 0:
            return 0
        
        position_value = self.portfolio_value * 0.1
        return position_value / price
    
    def execute_trade(self, price: float, quantity: float, date: pd.Timestamp):
        
        if quantity == 0:
            return
            
        trade_value = abs(quantity * price)
        commission_cost = trade_value * self.commission
        
        if quantity > 0:
            self.portfolio_value -= (trade_value + commission_cost)
        else:
            self.portfolio_value += (trade_value - commission_cost)
            
        self.trades.append({
            'date': date,
            'action': 'BUY' if quantity > 0 else 'SELL',
            'quantity': abs(quantity),
            'price': price,
            'value': trade_value,
            'commission': commission_cost
        })
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        
        signals = self.generate_signals(data)
        
        current_position = 0
        
        for i, (date, row) in enumerate(data.iterrows()):
            price = row['Close']
            signal = signals.iloc[i] if i < len(signals) else 0
            
            if signal == 1 and current_position <= 0:
                new_position = self.calculate_position_size(price, signal)
                trade_quantity = new_position - current_position
            elif signal == -1 and current_position >= 0:
                trade_quantity = -current_position
                new_position = 0
            else:
                trade_quantity = 0
                new_position = current_position
            
            if trade_quantity != 0:
                self.execute_trade(price, trade_quantity, date)
                current_position = new_position
            
            portfolio_value = self.portfolio_value + current_position * price
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'position': current_position,
                'price': price
            })
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        profitable_trades = [t for t in self.trades if t['action'] == 'SELL']
        win_rate = 0
        if profitable_trades:
            win_rate = 0.5
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'portfolio_history': portfolio_df
        }

class SMAStrategy(TradingStrategy):
    
    
    def __init__(self, short_window: int = 10, long_window: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        
        short_sma = data['Close'].rolling(window=self.short_window).mean()
        long_sma = data['Close'].rolling(window=self.long_window).mean()
        
        signals = pd.Series(0, index=data.index)
        
        signals[short_sma > long_sma] = 1
        
        signals[short_sma <= long_sma] = -1
        
        return signals

class RSIStrategy(TradingStrategy):
    
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        
        rsi = self.calculate_rsi(data['Close'])
        signals = pd.Series(0, index=data.index)
        
        signals[rsi < self.oversold] = 1
        
        signals[rsi > self.overbought] = -1
        
        return signals

class MLStrategy(TradingStrategy):
    
    
    def __init__(self, model, threshold: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        
        feature_cols = [col for col in data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
        X = data[feature_cols].fillna(method='ffill').fillna(0)
        
        try:
            predictions = self.model.predict(X)
            
            current_prices = data['Close'].values
            expected_returns = (predictions - current_prices) / current_prices
            
            signals = pd.Series(0, index=data.index)
            
            signals[expected_returns > self.threshold] = 1
            signals[expected_returns < -self.threshold] = -1
            
            return signals
            
        except Exception as e:
            print(f"Error in ML strategy: {e}")
            return pd.Series(0, index=data.index)

class MeanReversionStrategy(TradingStrategy):
    
    
    def __init__(self, window: int = 20, num_std: float = 2, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        
        close = data['Close']
        rolling_mean = close.rolling(window=self.window).mean()
        rolling_std = close.rolling(window=self.window).std()
        
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        
        signals = pd.Series(0, index=data.index)
        
        signals[close <= lower_band] = 1
        
        signals[close >= upper_band] = -1
        
        return signals

def compare_strategies(data: pd.DataFrame, strategies: List[TradingStrategy]) -> pd.DataFrame:
    
    results = []
    
    for i, strategy in enumerate(strategies):
        try:
            strategy_copy = strategy.__class__(**strategy.__dict__)
            performance = strategy_copy.backtest(data.copy())
            
            results.append({
                'strategy': strategy.__class__.__name__,
                'total_return': performance.get('total_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'total_trades': performance.get('total_trades', 0),
                'final_value': performance.get('final_value', 0)
            })
        except Exception as e:
            print(f"Error for strategy {strategy.__class__.__name__}: {e}")
            results.append({
                'strategy': strategy.__class__.__name__,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'final_value': 0
            })
    
    return pd.DataFrame(results)

def optimize_strategy_parameters(strategy_class, data: pd.DataFrame, param_grid: Dict) -> Dict:
    
    best_performance = -np.inf
    best_params = None
    
    import itertools
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for param_combination in itertools.product(*param_values):
        params = dict(zip(param_names, param_combination))
        
        try:
            strategy = strategy_class(**params)
            performance = strategy.backtest(data.copy())
            
            current_performance = performance.get('sharpe_ratio', -np.inf)
            
            if current_performance > best_performance:
                best_performance = current_performance
                best_params = params
                
        except Exception as e:
            print(f"Error for parameters {params}: {e}")
            continue
    
    return {
        'best_params': best_params,
        'best_performance': best_performance
    } 