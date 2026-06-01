import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

# Добавляем backend в путь
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelManager:
    def __init__(self):
        self.pipeline = None
        self.model_type = None
        self.is_loaded = False

    def load_model(self, file_path: str) -> bool:
        try:
            self.pipeline = joblib.load(file_path)
            self.is_loaded = True
            # Определяем тип модели
            if hasattr(self.pipeline, 'named_steps') and 'clf' in self.pipeline.named_steps:
                self.model_type = type(self.pipeline.named_steps['clf']).__name__
            else:
                self.model_type = type(self.pipeline).__name__
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False
            self.pipeline = None
            self.model_type = None
            return False

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_loaded or self.pipeline is None:
            raise ValueError("Модель не загружена")

        predictions = self.pipeline.predict(data)
        probabilities = self.pipeline.predict_proba(data)
        return predictions, probabilities

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "model_type": self.model_type
        }


# Глобальный экземпляр
model_manager = ModelManager()