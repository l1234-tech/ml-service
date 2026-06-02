import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.config import CATEGORICAL_COLS, NUMERIC_COLS  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402


class ModelManager:
    """Управление загрузкой и предсказанием модели."""

    def __init__(self):
        self._pipeline = None
        self._model_type = None
        self._is_loaded = False
        self._is_pipeline = False

    @property
    def is_loaded(self):
        return self._is_loaded

    @property
    def model_type(self):
        return self._model_type

    @property
    def pipeline(self):
        return self._pipeline

    def load_model(self, path: str) -> bool:
        """Загружает pipeline из .pkl файла."""
        try:
            self._pipeline = joblib.load(path)
            self._is_loaded = True
            self._is_pipeline = hasattr(self._pipeline, "steps")

            # Определяем тип модели по последнему шагу pipeline
            if self._is_pipeline:
                last_step = self._pipeline.steps[-1]
                if isinstance(last_step, tuple):
                    self._model_type = type(last_step[1]).__name__
                else:
                    self._model_type = type(last_step).__name__
            else:
                self._model_type = type(self._pipeline).__name__

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self._pipeline = None
            self._is_loaded = False
            self._model_type = None
            self._is_pipeline = False
            return False

    def _build_preprocessor(self):
        """Строит ColumnTransformer для предобработки сырых данных."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_COLS),
                ("cat", categorical_transformer, CATEGORICAL_COLS),
            ]
        )
        return preprocessor

    def _extract_classifier(self):
        """Извлекает классификатор из Pipeline или возвращает как есть."""
        if self._is_pipeline:
            steps_dict = dict(self._pipeline.steps)
            return steps_dict.get("clf", self._pipeline.steps[-1][1])
        return self._pipeline

    def predict(self, df: pd.DataFrame):
        """Делает предсказание и возвращает (preds, proba)."""
        if not self._is_loaded or self._pipeline is None:
            raise RuntimeError("Model is not loaded")

        try:
            if self._is_pipeline:
                # Пробуем использовать Pipeline как есть
                preds = self._pipeline.predict(df)
                proba = self._pipeline.predict_proba(df)
            else:
                # Только классификатор — применяем препроцессор вручную
                preprocessor = self._build_preprocessor()
                X = preprocessor.fit_transform(df)
                preds = self._pipeline.predict(X)
                proba = self._pipeline.predict_proba(X)
        except (ValueError, AttributeError) as e:
            # Если Pipeline не сработал (версионная несовместимость sklearn),
            # извлекаем классификатор и строим новый препроцессор
            print(f"[WARN] Pipeline failed, using fallback: {e}")
            classifier = self._extract_classifier()
            preprocessor = self._build_preprocessor()
            X = preprocessor.fit_transform(df)
            preds = classifier.predict(X)
            proba = classifier.predict_proba(X)

        return preds, proba

    def get_status(self) -> dict:
        """Возвращает статус модели."""
        return {
            "loaded": self._is_loaded,
            "model_type": self._model_type,
        }


# Глобальный экземпляр менеджера модели
model_manager = ModelManager()
# end of file
