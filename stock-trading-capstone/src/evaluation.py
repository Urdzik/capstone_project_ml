from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


class ModelEvaluator:

    def __init__(self):
        self.results = {}

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "model") -> Dict:

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        self.results[model_name] = metrics
        return metrics

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "model") -> Dict:

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

        self.results[model_name] = metrics
        return metrics

    def calculate_trading_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:

        if returns.empty:
            return {}

        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)

        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        win_rate = (returns > 0).mean()

        var_5 = np.percentile(returns, 5)

        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'var_5': var_5,
            'sortino_ratio': sortino_ratio
        }

        if benchmark_returns is not None:
            benchmark_total_return = (benchmark_returns + 1).prod() - 1
            excess_return = total_return - benchmark_total_return

            tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            risk_free_rate = 0.02
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))

            metrics.update({
                'excess_return': excess_return,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'tracking_error': tracking_error
            })

        return metrics

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = "Model", dates: pd.DatetimeIndex = None):

        plt.figure(figsize=(12, 6))

        if dates is not None:
            plt.plot(dates, y_true, label='Actual', alpha=0.7)
            plt.plot(dates, y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Date')
        else:
            plt.plot(y_true, label='Actual', alpha=0.7)
            plt.plot(y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Time')

        plt.ylabel('Price')
        plt.title(f'{model_name}: Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "Model"):

        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name}: Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(residuals, bins=50, alpha=0.7, density=True)
        axes[1].axvline(x=0, color='red', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'{model_name}: Residuals Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Model", class_names: List[str] = None):

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name}: Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

    def compare_models(self, models_results: Dict) -> pd.DataFrame:

        comparison_df = pd.DataFrame(models_results).T

        if 'r2' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('r2', ascending=False)
        elif 'accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('accuracy', ascending=False)

        return comparison_df

    def generate_report(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                        model_type: str = "regression", returns: pd.Series = None) -> str:

        report = f"=== Звіт для моделі: {model_name} ===\n\n"

        if model_type == "regression":
            metrics = self.evaluate_regression(y_true, y_pred, model_name)
            report += f"MSE: {metrics['mse']:.4f}\n"
            report += f"MAE: {metrics['mae']:.4f}\n"
            report += f"RMSE: {metrics['rmse']:.4f}\n"
            report += f"R²: {metrics['r2']:.4f}\n"
            report += f"MAPE: {metrics['mape']:.2f}%\n\n"

        elif model_type == "classification":
            metrics = self.evaluate_classification(y_true, y_pred, model_name)
            report += f"Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"Precision: {metrics['precision']:.4f}\n"
            report += f"Recall: {metrics['recall']:.4f}\n"
            report += f"F1-Score: {metrics['f1']:.4f}\n\n"

        if returns is not None:
            trading_metrics = self.calculate_trading_metrics(returns)
            report += "=== Торгові метрики ===\n"
            report += f"Загальна доходність: {trading_metrics['total_return']:.2%}\n"
            report += f"Річна доходність: {trading_metrics['annualized_return']:.2%}\n"
            report += f"Волатильність: {trading_metrics['volatility']:.2%}\n"
            report += f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}\n"
            report += f"Max Drawdown: {trading_metrics['max_drawdown']:.2%}\n"
            report += f"Win Rate: {trading_metrics['win_rate']:.2%}\n"

        return report


def backtest_model_predictions(predictions: np.ndarray, actual_prices: np.ndarray,
                               dates: pd.DatetimeIndex, initial_capital: float = 100000,
                               commission: float = 0.001) -> Dict:
    price_changes = np.diff(actual_prices) / actual_prices[:-1]
    pred_changes = np.diff(predictions) / predictions[:-1]

    signals = np.where(pred_changes > 0.01, 1, np.where(pred_changes < -0.01, -1, 0))

    returns = signals[:-1] * price_changes[1:]  # Затримка на 1 день

    trade_mask = np.abs(np.diff(np.concatenate([[0], signals]))) > 0
    commission_cost = trade_mask[:-1] * commission
    returns = returns - commission_cost

    portfolio_value = initial_capital * (1 + returns).cumprod()

    results_df = pd.DataFrame({
        'date': dates[2:],
        'actual_price': actual_prices[2:],
        'predicted_price': predictions[2:],
        'signal': signals[:-1],
        'return': returns,
        'portfolio_value': portfolio_value
    })

    evaluator = ModelEvaluator()
    trading_metrics = evaluator.calculate_trading_metrics(pd.Series(returns))

    return {
        'results_df': results_df,
        'trading_metrics': trading_metrics,
        'final_value': portfolio_value[-1] if len(portfolio_value) > 0 else initial_capital
    }


def calculate_feature_importance_metrics(X: pd.DataFrame, y: np.ndarray,
                                         feature_importance: np.ndarray) -> Dict:
    feature_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    feature_df['cumulative_importance'] = feature_df['importance'].cumsum()

    features_for_80_percent = (feature_df['cumulative_importance'] <= 0.8).sum()

    return {
        'feature_importance_df': feature_df,
        'top_features': feature_df.head(10)['feature'].tolist(),
        'features_for_80_percent': features_for_80_percent,
        'importance_concentration': feature_df.head(5)['importance'].sum()
    }
