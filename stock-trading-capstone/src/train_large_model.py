import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

warnings.filterwarnings('ignore')

class LargeModelTrainer:
    """
    Клас для навчання великих та ансамблевих моделей
    """
    
    def __init__(self, models_dir: str = "../models", random_state: int = 42):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.trained_models = {}
        self.best_model = None
        self.scaler = None
        
    def create_base_models(self) -> Dict[str, Any]:
        """
        Створює базові моделі для ансамблю
        """
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        return models
    
    def create_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Створює сітки гіперпараметрів для оптимізації
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.8, 0.9, 1.0]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            },
            'svr': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        return param_grids
    
    def preprocess_data(self, X: pd.DataFrame, y: np.ndarray, 
                       scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Попередня обробка даних
        """
        # Заповнення пропущених значень
        X_processed = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Масштабування ознак
        if scale_features:
            if self.scaler is None:
                self.scaler = RobustScaler()
                X_scaled = self.scaler.fit_transform(X_processed)
            else:
                X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = X_processed.values
        
        return X_scaled, y
    
    def optimize_hyperparameters(self, model_name: str, model: Any, param_grid: Dict,
                                X: np.ndarray, y: np.ndarray, cv: int = 5) -> Any:
        """
        Оптимізує гіперпараметри моделі
        """
        print(f"Оптимізація гіперпараметрів для {model_name}...")
        
        # Використовуємо TimeSeriesSplit для часових рядів
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Вибираємо тип пошуку залежно від розміру сітки
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        
        if total_combinations > 50:
            # Випадковий пошук для великих сіток
            search = RandomizedSearchCV(
                model, param_grid, n_iter=50, cv=tscv,
                scoring='neg_mean_squared_error', n_jobs=-1,
                random_state=self.random_state, verbose=1
            )
        else:
            # Повний пошук для малих сіток
            search = GridSearchCV(
                model, param_grid, cv=tscv,
                scoring='neg_mean_squared_error', n_jobs=-1,
                verbose=1
            )
        
        search.fit(X, y)
        print(f"Найкращі параметри для {model_name}: {search.best_params_}")
        print(f"Найкращий score: {-search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def train_individual_models(self, X: pd.DataFrame, y: np.ndarray, 
                              optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Навчає окремі моделі
        """
        X_processed, y_processed = self.preprocess_data(X, y)
        
        base_models = self.create_base_models()
        param_grids = self.create_hyperparameter_grids()
        
        trained_models = {}
        
        for model_name, model in base_models.items():
            print(f"\nНавчання моделі: {model_name}")
            
            try:
                if optimize_hyperparams and model_name in param_grids:
                    optimized_model = self.optimize_hyperparameters(
                        model_name, model, param_grids[model_name], 
                        X_processed, y_processed
                    )
                    trained_models[model_name] = optimized_model
                else:
                    model.fit(X_processed, y_processed)
                    trained_models[model_name] = model
                    
                print(f"Модель {model_name} успішно навчена")
                
            except Exception as e:
                print(f"Помилка навчання моделі {model_name}: {e}")
                continue
        
        self.trained_models = trained_models
        return trained_models
    
    def create_voting_ensemble(self, base_models: Dict[str, Any]) -> VotingRegressor:
        """
        Створює ансамбль голосування
        """
        # Беремо тільки успішно навчені моделі
        valid_models = [(name, model) for name, model in base_models.items() 
                       if model is not None]
        
        if len(valid_models) < 2:
            raise ValueError("Потрібно принаймні 2 моделі для ансамблю")
        
        voting_ensemble = VotingRegressor(
            estimators=valid_models,
            n_jobs=-1
        )
        
        return voting_ensemble
    
    def create_stacking_ensemble(self, base_models: Dict[str, Any], 
                               meta_learner: Any = None) -> StackingRegressor:
        """
        Створює стекінг ансамбль
        """
        if meta_learner is None:
            meta_learner = Ridge(alpha=1.0)
        
        # Беремо тільки успішно навчені моделі
        valid_models = [(name, model) for name, model in base_models.items() 
                       if model is not None]
        
        if len(valid_models) < 2:
            raise ValueError("Потрібно принаймні 2 моделі для ансамблю")
        
        stacking_ensemble = StackingRegressor(
            estimators=valid_models,
            final_estimator=meta_learner,
            cv=TimeSeriesSplit(n_splits=5),
            n_jobs=-1
        )
        
        return stacking_ensemble
    
    def train_ensemble_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Навчає ансамблеві моделі
        """
        if not self.trained_models:
            print("Спочатку навчіть базові моделі")
            return {}
        
        X_processed, y_processed = self.preprocess_data(X, y, scale_features=False)
        ensemble_models = {}
        
        try:
            # Voting Ensemble
            print("\nНавчання Voting Ensemble...")
            voting_ensemble = self.create_voting_ensemble(self.trained_models)
            voting_ensemble.fit(X_processed, y_processed)
            ensemble_models['voting_ensemble'] = voting_ensemble
            print("Voting Ensemble успішно навчений")
            
        except Exception as e:
            print(f"Помилка навчання Voting Ensemble: {e}")
        
        try:
            # Stacking Ensemble
            print("\nНавчання Stacking Ensemble...")
            stacking_ensemble = self.create_stacking_ensemble(self.trained_models)
            stacking_ensemble.fit(X_processed, y_processed)
            ensemble_models['stacking_ensemble'] = stacking_ensemble
            print("Stacking Ensemble успішно навчений")
            
        except Exception as e:
            print(f"Помилка навчання Stacking Ensemble: {e}")
        
        return ensemble_models
    
    def evaluate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                       y_test: np.ndarray) -> pd.DataFrame:
        """
        Оцінює всі моделі
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        X_test_processed, _ = self.preprocess_data(X_test, y_test, scale_features=False)
        
        results = []
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test_processed)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'model': model_name,
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'r2': r2
                })
                
            except Exception as e:
                print(f"Помилка оцінки моделі {model_name}: {e}")
                continue
        
        results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
        return results_df
    
    def select_best_model(self, all_models: Dict[str, Any], X_test: pd.DataFrame, 
                         y_test: np.ndarray) -> Tuple[str, Any]:
        """
        Вибирає найкращу модель
        """
        results_df = self.evaluate_models(all_models, X_test, y_test)
        
        if results_df.empty:
            return None, None
        
        best_model_name = results_df.iloc[0]['model']
        best_model = all_models[best_model_name]
        
        self.best_model = best_model
        
        print(f"\nНайкраща модель: {best_model_name}")
        print(f"R²: {results_df.iloc[0]['r2']:.4f}")
        print(f"RMSE: {results_df.iloc[0]['rmse']:.4f}")
        
        return best_model_name, best_model
    
    def save_models(self, models: Dict[str, Any], model_prefix: str = "large_model"):
        """
        Зберігає навчені моделі
        """
        for model_name, model in models.items():
            try:
                filename = f"{model_prefix}_{model_name}.joblib"
                filepath = self.models_dir / filename
                joblib.dump(model, filepath)
                print(f"Модель {model_name} збережена: {filepath}")
                
            except Exception as e:
                print(f"Помилка збереження моделі {model_name}: {e}")
        
        # Зберігаємо скейлер
        if self.scaler:
            scaler_path = self.models_dir / f"{model_prefix}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            print(f"Скейлер збережений: {scaler_path}")
    
    def train_complete_pipeline(self, X_train: pd.DataFrame, y_train: np.ndarray,
                              X_test: pd.DataFrame, y_test: np.ndarray,
                              optimize_hyperparams: bool = True) -> Dict:
        """
        Повний пайплайн навчання
        """
        print("=== Початок навчання великих моделей ===")
        
        # 1. Навчання базових моделей
        print("\n1. Навчання базових моделей...")
        individual_models = self.train_individual_models(
            X_train, y_train, optimize_hyperparams
        )
        
        # 2. Навчання ансамблевих моделей
        print("\n2. Навчання ансамблевих моделей...")
        ensemble_models = self.train_ensemble_models(X_train, y_train)
        
        # 3. Об'єднання всіх моделей
        all_models = {**individual_models, **ensemble_models}
        
        # 4. Оцінка моделей
        print("\n3. Оцінка моделей...")
        results_df = self.evaluate_models(all_models, X_test, y_test)
        print("\nРезультати оцінки:")
        print(results_df)
        
        # 5. Вибір найкращої моделі
        print("\n4. Вибір найкращої моделі...")
        best_model_name, best_model = self.select_best_model(all_models, X_test, y_test)
        
        # 6. Збереження моделей
        print("\n5. Збереження моделей...")
        self.save_models(all_models)
        
        return {
            'individual_models': individual_models,
            'ensemble_models': ensemble_models,
            'all_models': all_models,
            'results_df': results_df,
            'best_model_name': best_model_name,
            'best_model': best_model
        }

def load_large_model(model_name: str, models_dir: str = "../models"):
    """
    Завантажує збережену велику модель
    """
    models_path = Path(models_dir)
    model_file = models_path / f"large_model_{model_name}.joblib"
    scaler_file = models_path / "large_model_scaler.joblib"
    
    try:
        model = joblib.load(model_file)
        scaler = None
        
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
            
        print(f"Модель {model_name} завантажена успішно")
        return model, scaler
        
    except Exception as e:
        print(f"Помилка завантаження моделі {model_name}: {e}")
        return None, None 