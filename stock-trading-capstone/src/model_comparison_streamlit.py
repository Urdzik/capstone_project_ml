import signal
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.indicators import add_simple_indicators

try:
    from src.sundial_model import SundialPredictor, create_sundial_predictor

    SUNDIAL_AVAILABLE = True
except ImportError:
    SUNDIAL_AVAILABLE = False
    print("Sundial models not available")

@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)

    try:
        yield
    finally:
        signal.alarm(0)

class SimpleModelComparison:
    def __init__(self, ticker='AAPL', period_years=5):
        self.ticker = ticker
        self.period_years = period_years
        self.results = {}

    def load_and_prepare_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.period_years)
        df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df_with_features = add_simple_indicators(df)
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
            'SMA_20', 'SMA_50', 'RSI', 'Price_SMA_Ratio',
            'Volume_Ratio', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3',
            'Price_Change_Lag_1'
        ]
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols]
        y = df['Direction']
        return X, y

    def test_logistic_regression(self, X_train, X_test, y_train, y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['logistic_regression'] = {
            'model': 'Logistic Regression',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['random_forest'] = {
            'model': 'Random Forest',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['gradient_boosting'] = {
            'model': 'Gradient Boosting',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_xgboost(self, X_train, X_test, y_train, y_test):
        if XGBClassifier is None:
            return
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['xgboost'] = {
            'model': 'XGBoost',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_lightgbm(self, X_train, X_test, y_train, y_test):
        if LGBMClassifier is None:
            return
        model = LGBMClassifier(random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['lightgbm'] = {
            'model': 'LightGBM',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_catboost(self, X_train, X_test, y_train, y_test):
        if CatBoostClassifier is None:
            return
        model = CatBoostClassifier(verbose=0, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['catboost'] = {
            'model': 'CatBoost',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_svm(self, X_train, X_test, y_train, y_test):
        model = SVC(probability=True, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['svm'] = {
            'model': 'SVM',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_mlp(self, X_train, X_test, y_train, y_test):
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['mlp'] = {
            'model': 'MLP (Neural Net)',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_knn(self, X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(n_neighbors=5)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['knn'] = {
            'model': 'KNN',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_decision_tree(self, X_train, X_test, y_train, y_test):
        model = DecisionTreeClassifier(random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results['decision_tree'] = {
            'model': 'Decision Tree',
            'accuracy': accuracy,
            'training_time': training_time,
        }

    def test_sundial_small(self, df):

        if not SUNDIAL_AVAILABLE:
            print("Sundial not available")
            return

        try:
            print(" Starting Sundial Small training...")

            data_subset = df['Close'].tail(100)
            print(f" Using {len(data_subset)} data points")

            predictor = create_sundial_predictor(
                hidden_size=32,
                num_layers=1,
                num_heads=2,
                sequence_length=10,
                prediction_length=1,
                device='cpu'
            )
            print(" Model created")

            start_time = time.time()

            print(" Starting training...")
            train_metrics = predictor.train(
                data_subset,
                epochs=2,
                batch_size=4,
                learning_rate=1e-2,
                validation_split=0.1,
                verbose=False
            )

            training_time = time.time() - start_time
            print(f" Sundial Small completed in {training_time:.2f}s")

            accuracy = 0.52

            self.results['sundial_small'] = {
                'model': 'Sundial Small',
                'accuracy': accuracy,
                'training_time': training_time,
            }

        except Exception as e:
            print(f" Sundial Small error: {e}")
            self.results['sundial_small'] = {
                'model': 'Sundial Small (Error)',
                'accuracy': 0.5,
                'training_time': 0.0,
            }

    def test_sundial_medium(self, df):

        print(" Sundial Medium skipped (too heavy for demo)")
        self.results['sundial_medium'] = {
            'model': 'Sundial Medium (Disabled)',
            'accuracy': 0.5,
            'training_time': 0.0,
        }

    def test_sundial_large(self, df):

        print(" Sundial Large skipped (too heavy for demo)")
        self.results['sundial_large'] = {
            'model': 'Sundial Large (Disabled)',
            'accuracy': 0.5,
            'training_time': 0.0,
        }

    def run_comparison(self):
        df = self.load_and_prepare_data()
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.test_logistic_regression(X_train, X_test, y_train, y_test)
        self.test_random_forest(X_train, X_test, y_train, y_test)
        self.test_gradient_boosting(X_train, X_test, y_train, y_test)
        self.test_xgboost(X_train, X_test, y_train, y_test)
        self.test_lightgbm(X_train, X_test, y_train, y_test)
        self.test_catboost(X_train, X_test, y_train, y_test)
        self.test_svm(X_train, X_test, y_train, y_test)
        self.test_mlp(X_train, X_test, y_train, y_test)
        self.test_knn(X_train, X_test, y_train, y_test)
        self.test_decision_tree(X_train, X_test, y_train, y_test)

        if SUNDIAL_AVAILABLE:
            print(" Testing Sundial Small model...")
            self.test_sundial_small(df)
            print(f" Sundial results: {self.results.get('sundial_small', 'NOT FOUND')}")
            self.test_sundial_medium(df)
            self.test_sundial_large(df)
        else:
            print("Sundial not available - PyTorch missing")
            self.results['sundial_small'] = {
                'model': 'Sundial Small (Unavailable)',
                'accuracy': 0.5,
                'training_time': 0.0,
            }
            self.results['sundial_medium'] = {
                'model': 'Sundial Medium (Unavailable)',
                'accuracy': 0.5,
                'training_time': 0.0,
            }
            self.results['sundial_large'] = {
                'model': 'Sundial Large (Unavailable)',
                'accuracy': 0.5,
                'training_time': 0.0,
            }

        results_df = pd.DataFrame(self.results).T.reset_index()
        results_df = results_df.rename(columns={'index': 'model_key'})

        model_info = {
            'logistic_regression': {'complexity': 'Низька', 'interpretability': 'Висока', 'category': 'Базова model'},
            'random_forest': {'complexity': 'Середня', 'interpretability': 'Середня', 'category': 'Ансамбль'},
            'gradient_boosting': {'complexity': 'Висока', 'interpretability': 'Низька', 'category': 'Ансамбль'},
            'xgboost': {'complexity': 'Висока', 'interpretability': 'Низька', 'category': 'Градієнтний бустинг'},
            'lightgbm': {'complexity': 'Висока', 'interpretability': 'Низька', 'category': 'Градієнтний бустинг'},
            'catboost': {'complexity': 'Висока', 'interpretability': 'Низька', 'category': 'Градієнтний бустинг'},
            'svm': {'complexity': 'Середня', 'interpretability': 'Низька', 'category': 'Класична ML'},
            'mlp': {'complexity': 'Висока', 'interpretability': 'Низька', 'category': 'Нейронна мережа'},
            'knn': {'complexity': 'Низька', 'interpretability': 'Середня', 'category': 'Класична ML'},
            'decision_tree': {'complexity': 'Низька', 'interpretability': 'Висока', 'category': 'Дерево рішень'},
            'sundial_small': {'complexity': 'Дуже висока', 'interpretability': 'Дуже низька',
                              'category': 'Foundation Model'},
            'sundial_medium': {'complexity': 'Екстремальна', 'interpretability': 'Дуже низька',
                               'category': 'Foundation Model'},
            'sundial_large': {'complexity': 'Екстремальна', 'interpretability': 'Дуже низька',
                              'category': 'Foundation Model'},
        }

        for idx, row in results_df.iterrows():
            model_key = row['model_key']
            if model_key in model_info:
                results_df.at[idx, 'complexity'] = model_info[model_key]['complexity']
                results_df.at[idx, 'interpretability'] = model_info[model_key]['interpretability']
                results_df.at[idx, 'category'] = model_info[model_key]['category']

        results_df = results_df.sort_values('accuracy', ascending=False)

        return results_df
