import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from ml.config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL  # noqa: E402


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Подготавливает признаки: заполнение пропусков, приведение типов."""
    df = df.copy()

    # Заполняем пропуски
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # Приводим типы
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    return df


def split_features_target(df: pd.DataFrame):
    """Разделяет датасет на X (признаки) и y (целевая переменная)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
# end of file