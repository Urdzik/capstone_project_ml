import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    Клас для відбору найкращих ознак для торгових моделей
    """
    
    def __init__(self, selection_method='mutual_info', n_features=10):
        """
        :param selection_method: 'mutual_info', 'f_test', 'rfe', 'lasso', 'random_forest'
        :param n_features: кількість ознак для відбору
        """
        self.selection_method = selection_method
        self.n_features = n_features
        self.selector = None
        self.selected_features = None
        
    def fit_transform(self, X, y):
        """
        Відбирає найкращі ознаки та трансформує дані
        """
        if self.selection_method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_regression, k=self.n_features)
        elif self.selection_method == 'f_test':
            self.selector = SelectKBest(score_func=f_regression, k=self.n_features)
        elif self.selection_method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            self.selector = RFE(estimator=estimator, n_features_to_select=self.n_features)
        elif self.selection_method == 'lasso':
            lasso = LassoCV(cv=5, random_state=42)
            self.selector = SelectFromModel(lasso, max_features=self.n_features)
        elif self.selection_method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            self.selector = SelectFromModel(rf, max_features=self.n_features)
        else:
            raise ValueError(f"Невідомий метод відбору: {self.selection_method}")
            
        # Підгоняємо селектор та трансформуємо дані
        X_selected = self.selector.fit_transform(X, y)
        
        # Зберігаємо імена вибраних ознак
        if hasattr(self.selector, 'get_support'):
            support = self.selector.get_support()
            self.selected_features = X.columns[support].tolist()
        else:
            # Для SelectFromModel без get_support
            self.selected_features = X.columns[:X_selected.shape[1]].tolist()
            
        return X_selected
    
    def transform(self, X):
        """
        Трансформує нові дані використовуючи вже підігнаний селектор
        """
        if self.selector is None:
            raise ValueError("Спочатку викличте fit_transform()")
        return self.selector.transform(X)
    
    def get_feature_importance(self, X, y):
        """
        Повертає важливість ознак для різних методів
        """
        if self.selection_method in ['mutual_info', 'f_test']:
            scores = self.selector.scores_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'score': scores
            }).sort_values('score', ascending=False)
        elif self.selection_method == 'rfe':
            rankings = self.selector.ranking_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'ranking': rankings
            }).sort_values('ranking')
        elif self.selection_method in ['lasso', 'random_forest']:
            if hasattr(self.selector.estimator_, 'feature_importances_'):
                importances = self.selector.estimator_.feature_importances_
            elif hasattr(self.selector.estimator_, 'coef_'):
                importances = np.abs(self.selector.estimator_.coef_)
            else:
                return pd.DataFrame()
                
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            return pd.DataFrame()
            
        return feature_importance

def compare_feature_selection_methods(X, y, methods=['mutual_info', 'f_test', 'rfe', 'lasso'], n_features=10):
    """
    Порівнює різні методи відбору ознак
    """
    results = {}
    
    for method in methods:
        try:
            selector = FeatureSelector(selection_method=method, n_features=n_features)
            X_selected = selector.fit_transform(X, y)
            
            results[method] = {
                'selected_features': selector.selected_features,
                'n_selected': X_selected.shape[1],
                'feature_importance': selector.get_feature_importance(X, y)
            }
        except Exception as e:
            print(f"Помилка для методу {method}: {e}")
            results[method] = None
            
    return results

def get_correlated_features(X, threshold=0.95):
    """
    Знаходить високо корельовані ознаки для видалення
    """
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_features = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > threshold)
    ]
    
    return high_corr_features

def remove_low_variance_features(X, threshold=0.01):
    """
    Видаляє ознаки з низькою варіацією
    """
    from sklearn.feature_selection import VarianceThreshold
    
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X[selected_features], selected_features 