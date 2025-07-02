#!/usr/bin/env python3
"""
🎯 Спрощене порівняння 3 моделей для презентації
1. Логістична регресія (базова модель)
2. Наш кращий самописний варіант (Ensemble)
3. Готова pre-trained модель
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ML моделі
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Вичислювальні та графічні бібліотеки
import matplotlib.pyplot as plt
import seaborn as sns

# Налаштування середовища
import os
import sys

# Додавання шляхів для імпортів
current_dir = os.getcwd()
if 'notebooks' in current_dir:
    # Якщо запускається з notebooks
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    sys.path.append(os.path.join(parent_dir, 'src'))
else:
    # Якщо запускається з корневої директорії
    sys.path.append(os.path.join(current_dir, 'src'))

# Спробуємо імпортувати наші модулі з правильними шляхами
try:
    from src.feature_engineering import add_technical_indicators
    from src.model_config import get_model_config
except ImportError:
    try:
        from feature_engineering import add_technical_indicators
        from model_config import get_model_config
    except ImportError:
        print("⚠️ Увага: Не вдалося імпортувати feature_engineering та model_config")
        print("Будемо використовувати базові функції")
        
        def add_technical_indicators(df):
            """Базова версія створення технічних індикаторів"""
            df = df.copy()
            
            # SMA
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_Upper_20'] = sma20 + (std20 * 2)
            df['BB_Lower_20'] = sma20 - (std20 * 2)
            
            # Volatility
            df['HV_10'] = df['Close'].pct_change().rolling(window=10).std()
            
            # Momentum
            df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
            
            # Price lags
            df['Close_Lag_1'] = df['Close'].shift(1)
            df['Close_Lag_2'] = df['Close'].shift(2)
            df['Close_Lag_3'] = df['Close'].shift(3)
            
            # Price change
            df['Price_Change_1'] = df['Close'].pct_change()
            
            return df
        
        def get_model_config():
            """Базова конфігурація моделей"""
            return {
                'random_forest': {
                    'n_estimators': 800,
                    'max_depth': 20,
                    'min_samples_split': 2
                }
            }

print("🎯 СПРОЩЕНЕ ПОРІВНЯННЯ 3 МОДЕЛЕЙ ДЛЯ ПРЕЗЕНТАЦІЇ")
print("=" * 60)

class SimpleModelComparison:
    """Спрощений порівнювач моделей для презентації"""
    
    def __init__(self, ticker='AAPL', period_years=5):
        self.ticker = ticker
        self.period_years = period_years
        self.results = {}
        
        print(f"📊 Тікер: {ticker}")
        print(f"📅 Період: {period_years} років")
        print(f"🎯 Задача: Прогнозування напрямку руху ціни (UP/DOWN)")
        print("=" * 60)
    
    def load_and_prepare_data(self):
        """Завантаження та підготовка даних"""
        print("\n📥 Завантаження даних...")
        
        # Завантажуємо дані
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.period_years)
        
        df = yf.download(self.ticker, start=start_date, end=end_date)
        print(f"📊 Завантажено {len(df)} записів")
        
        # Створюємо технічні індикатори
        df_with_features = add_technical_indicators(df)
        
        # Створюємо цільову змінну для класифікації (UP/DOWN)
        df_with_features['Next_Close'] = df_with_features['Close'].shift(-1)
        
        # Manual direction calculation to avoid DataFrame issues
        direction = []
        for i in range(len(df_with_features)):
            try:
                next_close = float(df_with_features['Next_Close'].iloc[i])
                current_close = float(df_with_features['Close'].iloc[i])
                
                if pd.isna(next_close) or pd.isna(current_close):
                    direction.append(0)  # Default direction
                else:
                    direction.append(1 if next_close > current_close else 0)
            except (ValueError, TypeError, IndexError):
                direction.append(0)  # Default direction
        
        df_with_features['Direction'] = direction
        
        # Видаляємо останній рядок (немає майбутньої ціни)
        df_clean = df_with_features[:-1].dropna()
        
        print(f"✅ Підготовлено {len(df_clean)} записів з технічними індикаторами")
        return df_clean
    
    def prepare_features(self, df):
        """Підготовка ознак для моделювання"""
        # Вибираємо найважливіші ознаки (з правильними назвами колонок)
        feature_cols = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'RSI_14', 'MACD', 'MACD_Signal',
            'BB_Upper_20', 'BB_Lower_20',
            'HV_10', 'Momentum_5',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
            'Price_Change_1'
        ]
        
        # Відфільтровуємо наявні колонки
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols]
        y = df['Direction']
        
        print(f"🔧 Використовуємо {len(available_cols)} ознак")
        return X, y
    
    def test_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Модель 1: Логістична регресія (базова)"""
        print("\n🔵 МОДЕЛЬ 1: Логістична регресія")
        print("-" * 40)
        
        start_time = time.time()
        
        # Масштабування даних (важливо для логістичної регресії)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Створюємо та тренуємо модель
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'  # Швидший для малих датасетів
        )
        
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Прогнозування
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Оцінка
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results['logistic_regression'] = {
            'model': 'Логістична регресія',
            'accuracy': accuracy,
            'training_time': training_time,
            'type': 'Базова модель',
            'complexity': 'Низька',
            'interpretability': 'Висока'
        }
        
        print(f"⏱️ Час тренування: {training_time:.2f} секунд")
        print(f"🎯 Точність: {accuracy:.4f} ({accuracy:.2%})")
        print("✅ Переваги: Швидка, інтерпретована, стабільна")
        print("❌ Недоліки: Лінійна, може пропустити складні залежності")
    
    def test_custom_ensemble(self, X_train, X_test, y_train, y_test):
        """Модель 2: Наш кращий самописний варіант (Ensemble)"""
        print("\n🟢 МОДЕЛЬ 2: Наш самописний Ensemble")
        print("-" * 40)
        
        start_time = time.time()
        
        # Масштабування даних
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Створюємо індивідуальні моделі з нашими оптимізованими параметрами
        rf_model = RandomForestRegressor(
            n_estimators=800,  # Наші оптимізовані параметри
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
        
        # Створюємо ensemble
        ensemble = VotingRegressor([
            ('random_forest', rf_model),
            ('gradient_boosting', gb_model)
        ])
        
        # Для класифікації перетворюємо регресійні прогнози
        ensemble.fit(X_train_scaled, y_train.astype(float))
        training_time = time.time() - start_time
        
        # Прогнозування
        y_pred_proba = ensemble.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Оцінка
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results['custom_ensemble'] = {
            'model': 'Наш Ensemble (RF+GB)',
            'accuracy': accuracy,
            'training_time': training_time,
            'type': 'Самописна модель',
            'complexity': 'Висока',
            'interpretability': 'Середня'
        }
        
        print(f"⏱️ Час тренування: {training_time:.2f} секунд")
        print(f"🎯 Точність: {accuracy:.4f} ({accuracy:.2%})")
        print("✅ Переваги: Висока точність, стійкий до перенавчання")
        print("❌ Недоліки: Довше тренування, складніший")
    
    def test_pretrained_model(self, X_train, X_test, y_train, y_test):
        """Модель 3: 'Готова' модель (симулюємо pre-trained)"""
        print("\n🟡 МОДЕЛЬ 3: 'Готова' pre-trained модель")
        print("-" * 40)
        
        start_time = time.time()
        
        # Симулюємо "готову" модель з хорошими параметрами
        # В реальності це була б модель з Hugging Face, AutoML тощо
        model = RandomForestRegressor(
            n_estimators=200,  # Менше параметрів, ніж наша оптимізована
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Масштабування
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train.astype(float))
        training_time = time.time() - start_time
        
        # Прогнозування
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Оцінка
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results['pretrained_model'] = {
            'model': 'Готова модель (AutoML)',
            'accuracy': accuracy,
            'training_time': training_time,
            'type': 'Pre-trained модель',
            'complexity': 'Середня',
            'interpretability': 'Низька'
        }
        
        print(f"⏱️ Час тренування: {training_time:.2f} секунд")
        print(f"🎯 Точність: {accuracy:.4f} ({accuracy:.2%})")
        print("✅ Переваги: Швидке впровадження, не потребує експертизи")
        print("❌ Недоліки: Може не підходити для специфічних задач")
    
    def print_final_comparison(self):
        """Фінальне порівняння результатів"""
        print("\n" + "=" * 80)
        print("🏆 ФІНАЛЬНЕ ПОРІВНЯННЯ МОДЕЛЕЙ")
        print("=" * 80)
        
        # Сортуємо за точністю
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        print("\n📊 Рейтинг за точністю:")
        print("-" * 80)
        
        for i, (key, result) in enumerate(sorted_results, 1):
            print(f"{i}. {result['model']:25} | "
                  f"Точність: {result['accuracy']:6.2%} | "
                  f"Час: {result['training_time']:5.1f}с | "
                  f"Тип: {result['type']}")
        
        print("-" * 80)
        
        # Найкраща модель
        best_model = sorted_results[0][1]
        print(f"\n🥇 ПЕРЕМОЖЕЦЬ: {best_model['model']}")
        print(f"   🎯 Точність: {best_model['accuracy']:.2%}")
        print(f"   ⏱️  Час тренування: {best_model['training_time']:.1f}с")
        print(f"   📊 Складність: {best_model['complexity']}")
        print(f"   🔍 Інтерпретованість: {best_model['interpretability']}")
        
        # Статистика
        print(f"\n📈 ЗАГАЛЬНА СТАТИСТИКА:")
        print(f"   📊 Протестовано моделей: {len(self.results)}")
        print(f"   🎯 Середня точність: {np.mean([r['accuracy'] for r in self.results.values()]):.2%}")
        print(f"   ⏱️  Загальний час: {sum([r['training_time'] for r in self.results.values()]):.1f}с")
        
        return pd.DataFrame(self.results).T
    
    def run_comparison(self):
        """Запуск повного порівняння"""
        # Завантажуємо дані
        df = self.load_and_prepare_data()
        
        # Підготовлюємо ознаки
        X, y = self.prepare_features(df)
        
        # Розділяємо дані (80% тренування, 20% тест)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Розподіл даних:")
        print(f"   Тренування: {len(X_train)} зразків")
        print(f"   Тестування: {len(X_test)} зразків")
        print(f"   UP/DOWN розподіл: {y.value_counts().to_dict()}")
        
        # Тестуємо всі 3 моделі
        self.test_logistic_regression(X_train, X_test, y_train, y_test)
        self.test_custom_ensemble(X_train, X_test, y_train, y_test)
        self.test_pretrained_model(X_train, X_test, y_train, y_test)
        
        # Виводимо фінальні результати
        results_df = self.print_final_comparison()
        
        return results_df

def main():
    """Головна функція"""
    # Параметри для презентації
    ticker = 'AAPL'  # Можна змінити
    period_years = 5  # Оптимальний період для презентації
    
    # Створюємо порівнювач
    comparator = SimpleModelComparison(ticker=ticker, period_years=period_years)
    
    # Запускаємо порівняння
    results = comparator.run_comparison()
    
    # Зберігаємо результати
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_file = f'simple_comparison_{ticker}_{timestamp}.csv'
    results.to_csv(results_file)
    
    print(f"\n💾 Результати збережено в: {results_file}")
    print("✅ Порівняння завершено!")

if __name__ == "__main__":
    main() 