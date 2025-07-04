import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional, Any
import joblib
from pathlib import Path

class BaseModel:
    def __init__(self, model_name: str = "BaseModel"):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                    exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        if exclude_cols is None:
            exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Adj Close']
        
        data_leakage_patterns = [
            '_Lag_', 'Close_Max_', 'Close_Min_', 'Close_Mean_', 'Close_Median_',
            'EMA_', 'SMA_', 'PSAR', 'BB_Middle_', 'VPT',
        ]
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols + [target_col]:
                continue
            if any(pattern in col for pattern in data_leakage_patterns):
                continue
            feature_cols.append(col)
        
        X = df[feature_cols].dropna()
        y = df.loc[X.index, target_col]
        
        print(f"Features: {len(feature_cols)}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return X_train.values, X_test.values if X_test is not None else None
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        
        return X_train_scaled, X_test_scaled
    
    def evaluate(self, X_or_y_true, y_test_or_y_pred) -> Dict:
        if (hasattr(X_or_y_true, 'shape') and len(X_or_y_true.shape) > 1) or \
           (isinstance(X_or_y_true, pd.DataFrame)):
            X_test = X_or_y_true
            y_test = y_test_or_y_pred
            y_pred = self.predict(X_test)
            y_true = y_test.values if isinstance(y_test, pd.Series) else y_test
        else:
            y_true = X_or_y_true
            y_pred = y_test_or_y_pred
            
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,

            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def evaluate_on_data(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        y_pred = self.predict(X_test)
        return self.evaluate(y_test.values, y_pred)

class LinearRegressionModel(BaseModel):
    def __init__(self, regularization: str = None, alpha: float = 1.0):
        super().__init__("LinearRegression")
        self.regularization = regularization
        self.alpha = alpha
        
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        elif regularization == 'elastic':
            self.model = ElasticNet(alpha=alpha)
        else:
            self.model = LinearRegression()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             scale_features: bool = True) -> Dict:
        if scale_features:
            X_train_scaled, _ = self.scale_features(X_train)
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        y_pred_train = self.predict(X_train)
        train_metrics = self.evaluate(y_train, y_pred_train)
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X)

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 random_state: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        y_pred_train = self.predict(X_train)
        train_metrics = self.evaluate(y_train, y_pred_train)
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class GradientBoostingModel(BaseModel):
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, random_state: int = 42):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        y_pred_train = self.predict(X_train)
        train_metrics = self.evaluate(y_train, y_pred_train)
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        return self.model.predict(X)

class LSTMPyTorchModel(nn.Module):
    
    def __init__(self, input_size: int = 1, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMPyTorchModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class NeuralNetworkTrainer:
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        
    def prepare_sequences(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
            targets.append(data[i])
            
        return np.array(sequences), np.array(targets)
    
    def create_data_loader(self, X: np.ndarray, y: np.ndarray, 
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              epochs: int = 100, learning_rate: float = 0.001, 
              patience: int = 10) -> Dict:
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None
        }
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in data_loader:
                outputs = self.model(batch_X)
                predictions.extend(outputs.squeeze().cpu().numpy())
        
        return np.array(predictions)

class LSTMModelWrapper(BaseModel):
    
    def __init__(self, sequence_length: int = 60, hidden_size: int = 50, 
                 num_layers: int = 2, dropout: float = 0.2, epochs: int = 50, 
                 batch_size: int = 32, learning_rate: float = 0.001):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.data_scaler = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                    exclude_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        data = df[target_col].values.reshape(-1, 1)
        
        self.data_stats = {
            'mean': np.mean(data),
            'std': np.std(data)
        }
        
        print(f" Статистики data: mean=${self.data_stats['mean']:.2f}, std=${self.data_stats['std']:.2f}")
        
        data_normalized = (data - self.data_stats['mean']) / self.data_stats['std']
        
        X, y = [], []
        for i in range(self.sequence_length, len(data_normalized)):
            X.append(data_normalized[i-self.sequence_length:i, 0])
            y.append(data_normalized[i, 0])
        

        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, epochs: int = None, batch_size: int = None) -> Dict:
        
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
            
        if isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
            X_seq, y_seq = X_train, y_train
            if self.data_scaler is None:
                self.data_scaler = MinMaxScaler()
                dummy_data = np.array([[0], [1]])
                self.data_scaler.fit(dummy_data)
        else:
            df_combined = X_train.copy()
            df_combined['Close'] = y_train
            X_seq, y_seq = self.prepare_data(df_combined)
        
        if len(X_seq) < self.sequence_length:
            raise ValueError(f"Not enough data for creating sequences. Need minimum {self.sequence_length} records.")
        
        val_size = max(1, int(0.2 * len(X_seq)))
        X_train_seq = X_seq[:-val_size]
        y_train_seq = y_seq[:-val_size]
        X_val_seq = X_seq[-val_size:]
        y_val_seq = y_seq[-val_size:]
        
        self.model = LSTMPyTorchModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.trainer = NeuralNetworkTrainer(self.model, device=str(self.device))
        
        if len(X_train_seq.shape) == 2:
            X_train_reshaped = X_train_seq.reshape(-1, self.sequence_length, 1)
            X_val_reshaped = X_val_seq.reshape(-1, self.sequence_length, 1)
        else:
            X_train_reshaped = X_train_seq
            X_val_reshaped = X_val_seq
            
        train_loader = self.trainer.create_data_loader(
            X_train_reshaped,
            y_train_seq,
            batch_size=min(self.batch_size, len(X_train_seq))
        )
        val_loader = self.trainer.create_data_loader(
            X_val_reshaped,
            y_val_seq,
            batch_size=min(self.batch_size, len(X_val_seq)),
            shuffle=False
        )
        
        history = self.trainer.train(
            train_loader, val_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            patience=5
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X) -> np.ndarray:
        
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        if isinstance(X, np.ndarray) and len(X.shape) >= 2:
            if len(X.shape) == 2:
                X_reshaped = X.reshape(-1, self.sequence_length, 1)
            else:
                X_reshaped = X
                
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                batch_size = min(32, len(X_reshaped))
                for i in range(0, len(X_reshaped), batch_size):
                    batch = X_reshaped[i:i+batch_size]
                    X_tensor = torch.FloatTensor(batch).to(self.device)
                    pred_scaled = self.model(X_tensor).cpu().numpy()
                    
                    if hasattr(self.data_scaler, 'inverse_transform'):
                        pred = self.data_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                        predictions.extend(pred.flatten())
                    else:
                        predictions.extend(pred_scaled.flatten())
            
            return np.array(predictions)
        
        else:
            if hasattr(X, 'columns') and 'Close' in X.columns:
                close_values = X['Close'].values
            elif hasattr(X, 'iloc'):
                close_values = X.iloc[:, 0].values
            else:
                close_values = X
                
            if len(close_values) < self.sequence_length:
                last_price = close_values[-1] if len(close_values) > 0 else 100.0
                return np.full(len(X), last_price)
            
            recent_values = close_values[-self.sequence_length:].reshape(-1, 1)
            recent_scaled = self.data_scaler.transform(recent_values)
            
            X_seq = recent_scaled.reshape(1, self.sequence_length, 1)
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq).to(self.device)
                pred_scaled = self.model(X_tensor).cpu().numpy()
            
            pred = self.data_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
            
            return np.full(len(X), pred[0, 0])

class EnsembleModel(BaseModel):
    
    def __init__(self, models: List[BaseModel], weights: List[float] = None):
        super().__init__("EnsembleModel")
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        
        train_metrics = {}
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            metrics = model.train(X_train, y_train)
            train_metrics[f"model_{i+1}_{model.model_name}"] = metrics
            
        self.is_fitted = True
        
        y_pred_train = self.predict(X_train)
        ensemble_metrics = self.evaluate(y_train, y_pred_train)
        train_metrics['ensemble'] = ensemble_metrics
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        if not self.is_fitted:
            raise ValueError("Ensemble not trained!")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred

def train_test_split_time_series(X, y, test_size: float = 0.2):
    
    n = len(X)
    split = int(n * (1 - test_size))
    
    if hasattr(X, 'iloc'):
        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]
    else:
        X_train = X[:split]
        X_test = X[split:]
        y_train = y[:split]
        y_test = y[split:]
    
    return X_train, X_test, y_train, y_test

def cross_validate_time_series(model: BaseModel, X: pd.DataFrame, y: pd.Series, 
                              cv_folds: int = 5) -> Dict:
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    scores = {
        'mse': [],
        'mae': [],
        'r2': []
    }
    
    for train_idx, val_idx in tscv.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model.train(X_train_fold, y_train_fold)
        
        y_pred_fold = model.predict(X_val_fold)
        
        fold_metrics = model.evaluate(y_val_fold, y_pred_fold)
        
        scores['mse'].append(fold_metrics['mse'])
        scores['mae'].append(fold_metrics['mae'])
        scores['r2'].append(fold_metrics['r2'])
    
    avg_scores = {
        'mse_mean': np.mean(scores['mse']),
        'mse_std': np.std(scores['mse']),
        'mae_mean': np.mean(scores['mae']),
        'mae_std': np.std(scores['mae']),
        'r2_mean': np.mean(scores['r2']),
        'r2_std': np.std(scores['r2'])
    }
    
    return avg_scores

def save_model(model: Any, filepath: str, metadata: Dict = None):
    
    model_path = Path(filepath)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, nn.Module):
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'metadata': metadata
        }, filepath)
    else:
        joblib.dump({
            'model': model,
            'metadata': metadata
        }, filepath)
    
    print(f"Model saved: {filepath}")

def load_model(filepath: str, model_class: nn.Module = None) -> Tuple[Any, Dict]:
    
    if filepath.endswith('.pth'):
        checkpoint = torch.load(filepath, map_location='cpu')
        if model_class:
            model = model_class
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Must specify model_class for PyTorch model")
        metadata = checkpoint.get('metadata', {})
    else:
        data = joblib.load(filepath)
        model = data['model']
        metadata = data.get('metadata', {})
    
    print(f"Model loaded: {filepath}")
    return model, metadata

LSTMModel = LSTMModelWrapper

class StrictLinearRegression(LinearRegressionModel):
    
    def __init__(self, regularization: str = 'ridge', alpha: float = 100):
        super().__init__(regularization=regularization, alpha=alpha)
        self.model_name = "StrictLinearRegression"
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                    exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        exclude_patterns = [
            target_col,
            'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits',
            
            'SMA_', 'EMA_',
            'BB_',
            'Close_',
            'High_', 'Low_',
            '_Lag_',
            'PSAR',
            'VPT',
            'MACD_Signal',
            'MACD_Histogram',
            'KST_Signal',
            'Aroon_',
            'CCI',
            'Ultimate_Oscillator',
        ]
        
        available_cols = [col for col in df.columns 
                         if not any(pattern in col for pattern in exclude_patterns)]
        
        if exclude_cols:
            available_cols = [col for col in available_cols if col not in exclude_cols]
        
        if len(available_cols) < 5:
            print(f"Few features remaining: {len(available_cols)}")
            print(f"   Available: {available_cols}")
        
        X = df[available_cols].copy()
        y = df[target_col].copy()
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        suspicious_features = correlations[correlations > 0.95].index.tolist()
        
        if suspicious_features:
            print(f" ВИКЛЮЧЕНО підозрілі features (корр > 0.95): {suspicious_features}")
            X = X.drop(columns=suspicious_features)
            
        print(f"Строго відфільтровано. Залишилось features: {X.shape[1]}")
        
        final_correlations = X.corrwith(y).abs().sort_values(ascending=False)
        max_corr = final_correlations.iloc[0] if len(final_correlations) > 0 else 0
        print(f"Максимальна кореляція з ціною: {max_corr:.3f}")
        
        if max_corr > 0.9:
            print("Still high correlation - possible data leakage")
        
        return X, y

class LSTMModelWrapper(BaseModel):
    
    def __init__(self, sequence_length: int = 30, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3, epochs: int = 20, 
                 batch_size: int = 16, learning_rate: float = 0.001):
        super().__init__("ImprovedLSTM")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.trainer = None
        self.data_stats = None
        self.is_fitted = False
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Close', 
                    exclude_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        data = df[target_col].values.reshape(-1, 1)
        
        self.data_stats = {
            'mean': np.mean(data),
            'std': np.std(data)
        }
        
        print(f" Статистики data: mean=${self.data_stats['mean']:.2f}, std=${self.data_stats['std']:.2f}")
        
        data_normalized = (data - self.data_stats['mean']) / self.data_stats['std']
        
        X, y = [], []
        for i in range(self.sequence_length, len(data_normalized)):
            X.append(data_normalized[i-self.sequence_length:i, 0])
            y.append(data_normalized[i, 0])
        
        print(f"Normalized data range: min={np.min(data_normalized):.3f}, max={np.max(data_normalized):.3f}")
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, epochs: int = None, batch_size: int = None) -> Dict:
        
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
            
        if isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
            X_seq, y_seq = X_train, y_train
        else:
            df_combined = X_train.copy()
            df_combined['Close'] = y_train
            X_seq, y_seq = self.prepare_data(df_combined)
        
        val_size = max(1, int(0.2 * len(X_seq)))
        X_train_seq = X_seq[:-val_size]
        y_train_seq = y_seq[:-val_size]
        X_val_seq = X_seq[-val_size:]
        y_val_seq = y_seq[-val_size:]
        
        self.model = LSTMPyTorchModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        
        self.model.apply(init_weights)
        
        self.trainer = NeuralNetworkTrainer(self.model, device=str(self.device))
        
        X_train_reshaped = X_train_seq.reshape(-1, self.sequence_length, 1)
        X_val_reshaped = X_val_seq.reshape(-1, self.sequence_length, 1)
            
        train_loader = self.trainer.create_data_loader(
            X_train_reshaped,
            y_train_seq,
            batch_size=min(self.batch_size, len(X_train_seq))
        )
        val_loader = self.trainer.create_data_loader(
            X_val_reshaped,
            y_val_seq,
            batch_size=min(self.batch_size, len(X_val_seq)),
            shuffle=False
        )
        
        history = self.trainer.train(
            train_loader, val_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            patience=8
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X) -> np.ndarray:
        
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        if isinstance(X, np.ndarray) and len(X.shape) >= 2:
            if len(X.shape) == 2:
                X_reshaped = X.reshape(-1, self.sequence_length, 1)
            else:
                X_reshaped = X
                
            self.model.eval()
            predictions_normalized = []
            
            with torch.no_grad():
                batch_size = min(32, len(X_reshaped))
                for i in range(0, len(X_reshaped), batch_size):
                    batch = X_reshaped[i:i+batch_size]
                    X_tensor = torch.FloatTensor(batch).to(self.device)
                    pred_normalized = self.model(X_tensor).cpu().numpy()
                    predictions_normalized.extend(pred_normalized.flatten())
            
            predictions_normalized = np.array(predictions_normalized)
            predictions = predictions_normalized * self.data_stats['std'] + self.data_stats['mean']
            return predictions
        
        else:
            if hasattr(X, 'columns') and 'Close' in X.columns:
                close_values = X['Close'].values
            else:
                close_values = X.iloc[:, 0].values if hasattr(X, 'iloc') else X
                
            if len(close_values) < self.sequence_length:
                last_price = close_values[-1] if len(close_values) > 0 else 100.0
                return np.full(len(X), last_price)
            
            recent_values = close_values[-self.sequence_length:].reshape(-1, 1)
            recent_normalized = (recent_values - self.data_stats['mean']) / self.data_stats['std']
            
            X_seq = recent_normalized.reshape(1, self.sequence_length, 1)
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_seq).to(self.device)
                pred_normalized = self.model(X_tensor).cpu().numpy()
            
            pred = pred_normalized * self.data_stats['std'] + self.data_stats['mean']
            
            return np.full(len(X), pred[0, 0])
    
    def evaluate(self, X_or_y_true, y_test_or_y_pred) -> Dict:
        
        if hasattr(X_or_y_true, 'shape') and len(X_or_y_true.shape) == 1:
            y_true_norm = X_or_y_true
            y_pred_norm = y_test_or_y_pred
            
            y_true = y_true_norm * self.data_stats['std'] + self.data_stats['mean']
            y_pred = y_pred_norm * self.data_stats['std'] + self.data_stats['mean']
        else:
            X_test = X_or_y_true
            y_test_norm = y_test_or_y_pred
            
            y_pred = self.predict(X_test)
            
            y_true = y_test_norm * self.data_stats['std'] + self.data_stats['mean']
        
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
            
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse, 'mse': mse,
            'MAE': mae, 'mae': mae,
            'RMSE': rmse, 'rmse': rmse,
            'R2': r2, 'r2': r2
        }