import pickle
import joblib
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from typing import Any, Dict, Optional, Union

warnings.filterwarnings('ignore')

class ModelLoader:
    """
    Клас для збереження та завантаження ML моделей
    """
    
    def __init__(self, models_dir: str = "../models"):
        """
        :param models_dir: директорія для збереження моделей
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def save_sklearn_model(self, model: Any, model_name: str, metadata: Dict = None):
        """
        Зберігає sklearn модель
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        try:
            # Зберігаємо модель
            joblib.dump(model, model_path)
            
            # Зберігаємо метадані
            if metadata:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                    
            print(f"Модель збережена: {model_path}")
            return True
            
        except Exception as e:
            print(f"Помилка збереження моделі {model_name}: {e}")
            return False
    
    def load_sklearn_model(self, model_name: str):
        """
        Завантажує sklearn модель
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Модель {model_name} не знайдена")
                
            # Завантажуємо модель
            model = joblib.load(model_path)
            
            # Завантажуємо метадані
            metadata = None
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
            print(f"Модель завантажена: {model_path}")
            return model, metadata
            
        except Exception as e:
            print(f"Помилка завантаження моделі {model_name}: {e}")
            return None, None
    
    def save_pytorch_model(self, model: torch.nn.Module, model_name: str, 
                          optimizer: torch.optim.Optimizer = None, 
                          epoch: int = None, metadata: Dict = None):
        """
        Зберігає PyTorch модель
        """
        model_path = self.models_dir / f"{model_name}.pth"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        try:
            # Підготовка збережених даних
            save_dict = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__
            }
            
            if optimizer:
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            if epoch:
                save_dict['epoch'] = epoch
                
            # Зберігаємо модель
            torch.save(save_dict, model_path)
            
            # Зберігаємо метадані
            if metadata:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                    
            print(f"PyTorch модель збережена: {model_path}")
            return True
            
        except Exception as e:
            print(f"Помилка збереження PyTorch моделі {model_name}: {e}")
            return False
    
    def load_pytorch_model(self, model_class: torch.nn.Module, model_name: str, 
                          device: str = 'cpu'):
        """
        Завантажує PyTorch модель
        """
        model_path = self.models_dir / f"{model_name}.pth"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"PyTorch модель {model_name} не знайдена")
                
            # Завантажуємо збережені дані
            checkpoint = torch.load(model_path, map_location=device)
            
            # Ініціалізуємо модель
            model = model_class
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Завантажуємо метадані
            metadata = None
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
            print(f"PyTorch модель завантажена: {model_path}")
            return model, checkpoint, metadata
            
        except Exception as e:
            print(f"Помилка завантаження PyTorch моделі {model_name}: {e}")
            return None, None, None
    
    def save_predictions(self, predictions: np.ndarray, model_name: str, 
                        test_dates: pd.DatetimeIndex = None):
        """
        Зберігає прогнози моделі
        """
        predictions_path = self.models_dir / f"{model_name}_predictions.csv"
        
        try:
            if test_dates is not None:
                df = pd.DataFrame({
                    'date': test_dates,
                    'prediction': predictions
                })
            else:
                df = pd.DataFrame({
                    'prediction': predictions
                })
                
            df.to_csv(predictions_path, index=False)
            print(f"Прогнози збережені: {predictions_path}")
            return True
            
        except Exception as e:
            print(f"Помилка збереження прогнозів {model_name}: {e}")
            return False
    
    def load_predictions(self, model_name: str):
        """
        Завантажує прогнози моделі
        """
        predictions_path = self.models_dir / f"{model_name}_predictions.csv"
        
        try:
            if not predictions_path.exists():
                raise FileNotFoundError(f"Прогнози {model_name} не знайдені")
                
            df = pd.read_csv(predictions_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
            print(f"Прогнози завантажені: {predictions_path}")
            return df
            
        except Exception as e:
            print(f"Помилка завантаження прогнозів {model_name}: {e}")
            return None
    
    def list_models(self):
        """
        Повертає список збережених моделей
        """
        models = []
        
        # Sklearn моделі
        for file_path in self.models_dir.glob("*.joblib"):
            models.append({
                'name': file_path.stem,
                'type': 'sklearn',
                'path': str(file_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            })
            
        # PyTorch моделі
        for file_path in self.models_dir.glob("*.pth"):
            models.append({
                'name': file_path.stem,
                'type': 'pytorch',
                'path': str(file_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            })
            
        return models
    
    def delete_model(self, model_name: str):
        """
        Видаляє модель та пов'язані файли
        """
        files_to_delete = [
            self.models_dir / f"{model_name}.joblib",
            self.models_dir / f"{model_name}.pth",
            self.models_dir / f"{model_name}_metadata.json",
            self.models_dir / f"{model_name}_predictions.csv"
        ]
        
        deleted_files = []
        for file_path in files_to_delete:
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))
                
        if deleted_files:
            print(f"Видалені файли для моделі {model_name}: {deleted_files}")
            return True
        else:
            print(f"Модель {model_name} не знайдена")
            return False

# Глобальний екземпляр для зручності
model_loader = ModelLoader()

# Функції для зручного використання
def save_model(model, model_name: str, metadata: Dict = None):
    """Зберігає модель (автоматично визначає тип)"""
    if hasattr(model, 'fit') and hasattr(model, 'predict'):
        # Sklearn модель
        return model_loader.save_sklearn_model(model, model_name, metadata)
    elif isinstance(model, torch.nn.Module):
        # PyTorch модель
        return model_loader.save_pytorch_model(model, model_name, metadata=metadata)
    else:
        print(f"Невідомий тип моделі: {type(model)}")
        return False

def load_model(model_name: str, model_class=None):
    """Завантажує модель (автоматично визначає тип)"""
    # Спробуємо sklearn
    model, metadata = model_loader.load_sklearn_model(model_name)
    if model is not None:
        return model, metadata
        
    # Спробуємо PyTorch
    if model_class:
        model, checkpoint, metadata = model_loader.load_pytorch_model(model_class, model_name)
        return model, metadata
        
    print(f"Не вдалося завантажити модель {model_name}")
    return None, None 