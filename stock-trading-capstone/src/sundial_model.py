"""
🌅 Sundial Foundation Model Integration

Модуль для інтеграції Sundial foundation моделі в торгову систему.
Sundial - це переможець бенчмарків GIFT-Eval та Time-Series-Library 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path
import math
import json

warnings.filterwarnings('ignore')

class SundialConfig:
    """
    Конфігурація для Sundial Foundation Model
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

class SimpleSundialModel(nn.Module):
    """
    Спрощена Sundial модель для прогнозування фінансових часових рядів
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        prediction_length: int = 1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prediction_length = prediction_length
        
        # Вхідна проекція
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Позиційні ембеддінги
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        
        # Transformer шари
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Вихідний шар
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_length)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, input_size)
            mask: Optional attention mask
        Returns:
            predictions: Tensor of shape (batch_size, prediction_length)
        """
        batch_size, seq_length, _ = x.shape
        
        # Вхідна проекція
        x = self.input_projection(x)  # (batch_size, seq_length, hidden_size)
        
        # Позиційні ембеддінги
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.position_embeddings(positions)
        
        x = x + pos_embeddings
        x = self.dropout(x)
        
        # Transformer
        if mask is not None:
            # Конвертуємо маску для трансформера
            mask = mask.bool()
            # Інвертуємо маску: True -> False (не маскувати), False -> True (маскувати)
            mask = ~mask
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Беремо останній timestep для прогнозування
        last_hidden = x[:, -1, :]  # (batch_size, hidden_size)
        
        # Генеруємо прогноз
        predictions = self.output_projection(last_hidden)  # (batch_size, prediction_length)
        
        return predictions

class FinancialTimeSeriesPreprocessor:
    """
    Препроцесор для фінансових часових рядів
    """
    def __init__(self, sequence_length: int = 60, prediction_length: int = 1):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler_mean = None
        self.scaler_std = None
        
    def fit(self, data: pd.Series):
        """
        Навчає препроцесор на даних
        """
        # Обчислюємо зміни цін у відсотках
        price_changes = data.pct_change().fillna(0) * 100
        
        self.scaler_mean = price_changes.mean()
        self.scaler_std = price_changes.std()
        
    def transform(self, data: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Перетворює дані на послідовності для навчання
        """
        # Обчислюємо зміни цін у відсотках
        price_changes = data.pct_change().fillna(0) * 100
        
        # Нормалізуємо
        if self.scaler_mean is not None and self.scaler_std is not None:
            price_changes = (price_changes - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(price_changes) - self.prediction_length + 1):
            # Вхідна послідовність
            seq = price_changes.iloc[i-self.sequence_length:i].values
            sequences.append(seq)
            
            # Цільове значення
            target = price_changes.iloc[i:i+self.prediction_length].values
            targets.append(target)
        
        # Конвертуємо в тензори
        X = torch.FloatTensor(sequences).unsqueeze(-1)  # (n_samples, seq_length, 1)
        y = torch.FloatTensor(targets)  # (n_samples, prediction_length)
        
        return X, y
    
    def inverse_transform(self, predictions: torch.Tensor) -> np.ndarray:
        """
        Зворотне перетворення прогнозів
        """
        predictions = predictions.detach().cpu().numpy()
        
        if self.scaler_mean is not None and self.scaler_std is not None:
            predictions = predictions * self.scaler_std + self.scaler_mean
        
        return predictions

class SundialPredictor:
    """
    Основний клас для використання Sundial моделі
    """
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        sequence_length: int = 60,
        prediction_length: int = 1,
        device: str = 'cpu'
    ):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.device = device
        
        self.model = SimpleSundialModel(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_length=sequence_length,
            prediction_length=prediction_length
        ).to(device)
        
        self.preprocessor = FinancialTimeSeriesPreprocessor(
            sequence_length=sequence_length,
            prediction_length=prediction_length
        )
        
        self.is_fitted = False
        
    def train(
        self,
        data: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        validation_split: float = 0.2,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Навчає модель
        """
        if verbose:
            print("Підготовка даних...")
        
        # Підготовляємо дані
        self.preprocessor.fit(data)
        X, y = self.preprocessor.transform(data)
        
        # Розділяємо на train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Створюємо DataLoader'и
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Налаштовуємо оптимізатор та функцію втрат
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Для early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        if verbose:
            print(f"Початок навчання на {len(X_train)} зразках...")
        
        for epoch in range(epochs):
            # Навчання
            self.model.train()
            epoch_train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Валідація
            self.model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping на епосі {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Епоха {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        self.is_fitted = True
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, data: pd.Series, return_sequences: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Робить прогнози
        """
        if not self.is_fitted:
            raise ValueError("Модель не навчена! Спочатку викличте train()")
        
        # Підготовляємо дані
        X, _ = self.preprocessor.transform(data)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), 32):  # Обробляємо батчами
                batch_X = X[i:i+32].to(self.device)
                batch_pred = self.model(batch_X)
                predictions.append(batch_pred.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        
        # Зворотне перетворення
        predictions = self.preprocessor.inverse_transform(predictions)
        
        if return_sequences:
            # Повертаємо також вхідні послідовності
            sequences = X.squeeze(-1).numpy()
            sequences = self.preprocessor.inverse_transform(torch.FloatTensor(sequences))
            return predictions, sequences
        
        return predictions
    
    def predict_next(self, data: pd.Series, n_steps: int = 1) -> np.ndarray:
        """
        Прогнозує наступні n кроків
        """
        if not self.is_fitted:
            raise ValueError("Модель не навчена! Спочатку викличте train()")
        
        # Беремо останні sequence_length точок
        recent_data = data.tail(self.sequence_length)
        
        predictions = []
        current_data = recent_data.copy()
        
        for _ in range(n_steps):
            # Робимо прогноз для поточних даних
            pred = self.predict(current_data)
            
            if len(pred) > 0:
                # Беремо останній прогноз
                next_pred = pred[-1]
                predictions.extend(next_pred if hasattr(next_pred, '__iter__') else [next_pred])
                
                # Оновлюємо дані для наступного прогнозу
                # Конвертуємо прогноз назад в ціну
                last_price = current_data.iloc[-1]
                predicted_change = next_pred[0] / 100  # Конвертуємо з відсотків
                next_price = last_price * (1 + predicted_change)
                
                # Додаємо нову ціну до даних
                new_data = pd.Series([next_price], index=[current_data.index[-1] + pd.Timedelta(days=1)])
                current_data = pd.concat([current_data, new_data])
                current_data = current_data.tail(self.sequence_length)
        
        return np.array(predictions)
    
    def evaluate(self, data: pd.Series) -> Dict:
        """
        Оцінює модель
        """
        predictions = self.predict(data)
        
        # Підготовляємо цільові значення
        X, y_true = self.preprocessor.transform(data)
        y_true = self.preprocessor.inverse_transform(y_true)
        
        # Обчислюємо метрики
        mse = np.mean((predictions - y_true) ** 2)
        mae = np.mean(np.abs(predictions - y_true))
        rmse = np.sqrt(mse)
        
        # Точність напрямку (чи правильно передбачили зростання/спадання)
        direction_accuracy = np.mean(np.sign(predictions.flatten()) == np.sign(y_true.flatten()))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }
    
    def save_model(self, filepath: str):
        """
        Зберігає модель
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'preprocessor_mean': self.preprocessor.scaler_mean,
            'preprocessor_std': self.preprocessor.scaler_std,
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'model_config': {
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
            },
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Модель збережена: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Завантажує модель
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Відновлюємо конфігурацію
        config = checkpoint['model_config']
        self.sequence_length = checkpoint['sequence_length']
        self.prediction_length = checkpoint['prediction_length']
        
        # Створюємо нову модель з правильними параметрами
        self.model = SimpleSundialModel(
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            max_seq_length=self.sequence_length,
            prediction_length=self.prediction_length
        ).to(self.device)
        
        # Завантажуємо ваги
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Відновлюємо препроцесор
        self.preprocessor = FinancialTimeSeriesPreprocessor(
            sequence_length=self.sequence_length,
            prediction_length=self.prediction_length
        )
        self.preprocessor.scaler_mean = checkpoint['preprocessor_mean']
        self.preprocessor.scaler_std = checkpoint['preprocessor_std']
        
        self.is_fitted = checkpoint['is_fitted']
        
        print(f"Модель завантажена: {filepath}")

# Функції для зручності використання
def create_sundial_predictor(
    hidden_size: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    sequence_length: int = 60,
    prediction_length: int = 1,
    device: str = 'cpu'
) -> SundialPredictor:
    """
    Створює Sundial predictor з заданими параметрами
    """
    return SundialPredictor(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        device=device
    )

def train_sundial_on_stock_data(
    stock_data: pd.DataFrame,
    price_column: str = 'Close',
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    sequence_length: int = 60,
    prediction_length: int = 1,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[SundialPredictor, Dict]:
    """
    Навчає Sundial модель на даних акцій
    """
    predictor = create_sundial_predictor(
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        device=device
    )
    
    price_series = stock_data[price_column]
    
    training_results = predictor.train(
        data=price_series,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose
    )
    
    return predictor, training_results

def load_and_predict(model_path: str, stock_data: pd.DataFrame, price_column: str = 'Close') -> np.ndarray:
    """
    Завантажує модель і робить прогнози
    """
    predictor = SundialPredictor()
    predictor.load_model(model_path)
    
    price_series = stock_data[price_column]
    predictions = predictor.predict(price_series)
    
    return predictions


class SundialTradingStrategy:
    """Торгова стратегія на базі Sundial прогнозів"""
    
    def __init__(self, predictor, initial_capital=10000, window_size=60, forecast_horizon=5):
        """
        Ініціалізація торгової стратегії
        
        Args:
            predictor (SundialPredictor): Екземпляр Sundial предиктора
            initial_capital (float): Початковий капітал
            window_size (int): Розмір вікна для історії
            forecast_horizon (int): Горизонт прогнозування
        """
        self.predictor = predictor
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
        # Стан портфеля
        self.cash = initial_capital
        self.shares = 0
        self.portfolio_value = []
        self.trades = []
        
    def make_prediction(self, price_history):
        """Робить прогноз ціни на наступні дні"""
        if len(price_history) < self.window_size:
            return None
            
        # Використовуємо останні window_size значень
        input_seq = price_history[-self.window_size:]
        
        # Робимо прогноз
        prediction = self.predictor.predict(input_seq, self.forecast_horizon)
        
        return prediction
    
    def generate_signal(self, current_price, prediction):
        """
        Генерує торговий сигнал на основі прогнозу
        
        Args:
            current_price (float): Поточна ціна
            prediction (np.array): Прогнозовані значення
            
        Returns:
            str: Торговий сигнал ('BUY', 'SELL', 'HOLD')
        """
        if prediction is None or len(prediction) == 0:
            return "HOLD"
        
        # Обчислюємо очікувану зміну ціни
        expected_price = prediction[-1]  # Ціна через forecast_horizon днів
        price_change_pct = (expected_price - current_price) / current_price
        
        # Торгові сигнали з порогами
        if price_change_pct > 0.02:  # Очікуємо зростання >2%
            return "BUY"
        elif price_change_pct < -0.02:  # Очікуємо падіння >2%
            return "SELL"
        else:
            return "HOLD"
    
    def execute_trade(self, signal, current_price, date):
        """Виконує торгову операцію"""
        if signal == "BUY" and self.cash > current_price:
            # Купуємо максимальну кількість акцій
            shares_to_buy = int(self.cash // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.cash -= cost
                self.shares += shares_to_buy
                self.trades.append({
                    'date': date,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost
                })
                
        elif signal == "SELL" and self.shares > 0:
            # Продаємо всі акції
            proceeds = self.shares * current_price
            self.cash += proceeds
            self.trades.append({
                'date': date,
                'action': 'SELL',
                'shares': self.shares,
                'price': current_price,
                'proceeds': proceeds
            })
            self.shares = 0
    
    def get_portfolio_value(self, current_price):
        """Розраховує поточну вартість портфеля"""
        return self.cash + (self.shares * current_price)
    
    def get_performance_metrics(self, final_value):
        """Розраховує метрики ефективності"""
        total_return = (final_value - self.initial_capital) / self.initial_capital
        return {
            'total_return': total_return,
            'final_value': final_value,
            'initial_capital': self.initial_capital,
            'total_trades': len(self.trades)
        }


def test_sundial_model():
    """Тестування Sundial моделі"""
    print("🧪 Тестування Sundial Foundation Model...")
    
    # Створюємо предиктор
    predictor = SundialPredictor()
    
    # Спробуємо завантажити модель
    success = predictor.load_model()
    
    if success:
        print("✅ Модель успішно завантажена!")
    else:
        print("⚠️ Модель не завантажена, буде використано fallback")
    
    # Тестові дані
    test_data = np.random.randn(60) * 10 + 200  # Симуляція цін акцій
    
    # Робимо прогноз
    prediction = predictor.predict(test_data, forecast_length=5)
    
    print(f"📊 Прогноз на 5 днів: {prediction}")
    print("🧪 Тест завершено!")


if __name__ == "__main__":
    test_sundial_model() 