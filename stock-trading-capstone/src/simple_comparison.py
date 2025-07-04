import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
import sys

current_dir = os.getcwd()
if 'notebooks' in current_dir:
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    sys.path.append(os.path.join(parent_dir, 'src'))
else:
    sys.path.append(os.path.join(current_dir, 'src'))

try:
    from src.feature_engineering import add_technical_indicators
    from src.model_config import get_model_config
except ImportError:
    try:
        from feature_engineering import add_technical_indicators
        from model_config import get_model_config
    except ImportError:
        print("Could not import feature_engineering та model_config")
        
        def add_technical_indicators(df):
            
            df = df.copy()
            
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_Upper_20'] = sma20 + (std20 * 2)
            df['BB_Lower_20'] = sma20 - (std20 * 2)
            
            df['HV_10'] = df['Close'].pct_change().rolling(window=10).std()
            
            df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
            
            df['Close_Lag_1'] = df['Close'].shift(1)
            df['Close_Lag_2'] = df['Close'].shift(2)
            df['Close_Lag_3'] = df['Close'].shift(3)
            
            df['Price_Change_1'] = df['Close'].pct_change()
            
            return df
        
        def get_model_config():
            
            return {
                'random_forest': {
                    'n_estimators': 800,
                    'max_depth': 20,
                    'min_samples_split': 2
                }
            }

class SimpleModelComparison:
    
    
    def __init__(self, ticker='AAPL', period_years=5):
        self.ticker = ticker
        self.period_years = period_years
        self.results = {}
    
    def load_and_prepare_data(self):
        

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.period_years)
        
        df = yf.download(self.ticker, start=start_date, end=end_date)

        df_with_features = add_technical_indicators(df)
        
        df_with_features['Next_Close'] = df_with_features['Close'].shift(-1)
        
        direction = []
        for i in range(len(df_with_features)):
            try:
                next_close = float(df_with_features['Next_Close'].iloc[i])
                current_close = float(df_with_features['Close'].iloc[i])
                
                if pd.isna(next_close) or pd.isna(current_close):
                    direction.append(0)
                else:
                    direction.append(1 if next_close > current_close else 0)
            except (ValueError, TypeError, IndexError):
                direction.append(0)
        
        df_with_features['Direction'] = direction
        
        df_clean = df_with_features[:-1].dropna()
        
        return df_clean
    
    def prepare_features(self, df):
        
        feature_cols = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'RSI_14', 'MACD', 'MACD_Signal',
            'BB_Upper_20', 'BB_Lower_20',
            'HV_10', 'Momentum_5',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
            'Price_Change_1'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols]
        y = df['Direction']
        
        return X, y
    
    def test_logistic_regression(self, X_train, X_test, y_train, y_test):

        start_time = time.time()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
        
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results['logistic_regression'] = {
            'model': 'Логістична регресія',
            'accuracy': accuracy,
            'training_time': training_time,
            'type': 'Базова model',
            'complexity': 'Низька',
            'interpretability': 'Висока'
        }
        
    
    def test_custom_ensemble(self, X_train, X_test, y_train, y_test):
        start_time = time.time()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestRegressor(
            n_estimators=800,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            random_state=42
        )
        
        ensemble = VotingRegressor([
            ('random_forest', rf_model),
            ('gradient_boosting', gb_model)
        ])
        
        ensemble.fit(X_train_scaled, y_train.astype(float))
        training_time = time.time() - start_time
        
        y_pred_proba = ensemble.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results['custom_ensemble'] = {
            'model': 'Наш Ensemble (RF+GB)',
            'accuracy': accuracy,
            'training_time': training_time,
            'type': 'Самописна model',
            'complexity': 'Висока',
            'interpretability': 'Середня'
        }

    
    def test_pretrained_model(self, X_train, X_test, y_train, y_test):
        
        start_time = time.time()
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train.astype(float))
        training_time = time.time() - start_time
        
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results['pretrained_model'] = {
            'model': 'Готова model (AutoML)',
            'accuracy': accuracy,
            'training_time': training_time,
            'type': 'Pre-trained model',
            'complexity': 'Середня',
            'interpretability': 'Низька'
        }
        

    
    def print_final_comparison(self):
        
        
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        print("\n Рейтинг за точністю:")
        
        for i, (key, result) in enumerate(sorted_results, 1):
            print(f"{i}. {result['model']:25} | "
                  f"Точність: {result['accuracy']:6.2%} | "
                  f"Час: {result['training_time']:5.1f}с | "
                  f"Тип: {result['type']}")
        
        
        best_model = sorted_results[0][1]
        print(f"\n ПЕРЕМОЖЕЦЬ: {best_model['model']}")
        print(f"    Точність: {best_model['accuracy']:.2%}")
        print(f"   Training time: {best_model['training_time']:.1f}с")
        print(f"    Складність: {best_model['complexity']}")
        print(f"    Інтерпретованість: {best_model['interpretability']}")
        
        print(f"\n ЗАГАЛЬНА СТАТИСТИКА:")
        print(f"    Протестовано моделей: {len(self.results)}")
        print(f"    Середня точність: {np.mean([r['accuracy'] for r in self.results.values()]):.2%}")
        print(f"   Total time: {sum([r['training_time'] for r in self.results.values()]):.1f}с")
        
        return pd.DataFrame(self.results).T
    
    def run_comparison(self):
        
        df = self.load_and_prepare_data()
        
        X, y = self.prepare_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        self.test_logistic_regression(X_train, X_test, y_train, y_test)
        self.test_custom_ensemble(X_train, X_test, y_train, y_test)
        self.test_pretrained_model(X_train, X_test, y_train, y_test)
        
        results_df = self.print_final_comparison()
        
        return results_df

def main():
    
    ticker = 'AAPL'
    period_years = 5
    
    comparator = SimpleModelComparison(ticker=ticker, period_years=period_years)
    
    results = comparator.run_comparison()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_file = f'simple_comparison_{ticker}_{timestamp}.csv'
    results.to_csv(results_file)
    

if __name__ == "__main__":
    main() 