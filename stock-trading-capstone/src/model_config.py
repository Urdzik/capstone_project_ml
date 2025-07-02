"""
Конфігурації для різних моделей машинного навчання
"""

# Конфігурації для традиційних ML моделей
LINEAR_REGRESSION_CONFIG = {
    'fit_intercept': True,
    'normalize': False,
    'copy_X': True,
    'n_jobs': None
}

RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'auto',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}

GRADIENT_BOOSTING_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'subsample': 1.0,
    'random_state': 42
}

XGB_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

LIGHTGBM_CONFIG = {
    'n_estimators': 100,
    'max_depth': -1,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'epsilon': 0.1
}

# Конфігурації для нейронних мереж
LSTM_CONFIG = {
    'sequence_length': 60,
    'input_size': 1,
    'hidden_size': 50,
    'num_layers': 2,
    'output_size': 1,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 10
}

GRU_CONFIG = {
    'sequence_length': 60,
    'input_size': 1,
    'hidden_size': 50,
    'num_layers': 2,
    'output_size': 1,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 10
}

TRANSFORMER_CONFIG = {
    'sequence_length': 60,
    'd_model': 64,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15
}

# Конфігурації для ансамблевих методів
VOTING_CLASSIFIER_CONFIG = {
    'voting': 'soft',
    'n_jobs': -1
}

STACKING_CONFIG = {
    'cv': 5,
    'n_jobs': -1,
    'passthrough': False
}

# Конфігурації для гіперпараметрів оптимізації
HYPERPARAMETER_SEARCH_CONFIG = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.1, 0.2],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0]
    }
}

# Конфігурації для обробки даних
DATA_PREPROCESSING_CONFIG = {
    'fill_method': 'forward',  # 'forward', 'backward', 'interpolate', 'drop'
    'scaling_method': 'standard',  # 'standard', 'minmax', 'robust', 'none'
    'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest', 'none'
    'outlier_threshold': 3,
    'feature_selection': {
        'method': 'mutual_info',
        'n_features': 15
    }
}

# Конфігурації для розбиття даних
TRAIN_TEST_SPLIT_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'time_series_split': True,
    'shuffle': False  # Для часових рядів завжди False
}

# Конфігурації для cross-validation
CROSS_VALIDATION_CONFIG = {
    'cv_folds': 5,
    'time_series_cv': True,
    'gap': 0,  # Розрив між train та test у TimeSeriesSplit
    'test_size': None  # Розмір тестового набору у TimeSeriesSplit
}

# Конфігурації для метрик оцінки
EVALUATION_METRICS_CONFIG = {
    'regression': ['mse', 'mae', 'rmse', 'r2', 'mape'],
    'classification': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'trading_specific': ['sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate']
}

# Конфігурації для торгових стратегій
TRADING_STRATEGY_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1%
    'slippage': 0.001,    # 0.1%
    'position_sizing': 'fixed_fractional',  # 'fixed', 'fixed_fractional', 'kelly'
    'risk_per_trade': 0.02,  # 2%
    'max_positions': 1,
    'stop_loss': 0.05,    # 5%
    'take_profit': 0.10   # 10%
}

# Функція для отримання конфігурації моделі
def get_model_config(model_name):
    """
    Повертає конфігурацію для вказаної моделі
    """
    configs = {
        'linear_regression': LINEAR_REGRESSION_CONFIG,
        'random_forest': RANDOM_FOREST_CONFIG,
        'gradient_boosting': GRADIENT_BOOSTING_CONFIG,
        'xgboost': XGB_CONFIG,
        'lightgbm': LIGHTGBM_CONFIG,
        'svm': SVM_CONFIG,
        'lstm': LSTM_CONFIG,
        'gru': GRU_CONFIG,
        'transformer': TRANSFORMER_CONFIG,
        'voting': VOTING_CLASSIFIER_CONFIG,
        'stacking': STACKING_CONFIG
    }
    
    return configs.get(model_name, {})

# Функція для оновлення конфігурації
def update_config(base_config, custom_config):
    """
    Оновлює базову конфігурацію кастомними параметрами
    """
    updated_config = base_config.copy()
    updated_config.update(custom_config)
    return updated_config 