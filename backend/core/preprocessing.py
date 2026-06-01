import pandas as pd
import numpy as np
from typing import Dict, Any


def create_dataframe_from_raw(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Преобразует сырые 13 полей в DataFrame для pipeline.
    Pipeline сам сделает one-hot encoding.
    """
    # Убеждаемся, что все нужные колонки есть
    required_cols = [
        'person_age', 'person_income', 'person_emp_exp', 'person_gender',
        'person_education', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ]

    # Проверяем наличие всех колонок
    missing = [col for col in required_cols if col not in data]
    if missing:
        raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing)}")

    # Создаём DataFrame с одной строкой
    df = pd.DataFrame([data])

    # Преобразуем числовые колонки
    numeric_cols = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def validate_csv_columns(df: pd.DataFrame, require_target: bool = False) -> None:
    """
    Проверяет, что CSV содержит все необходимые колонки.
    """
    required_cols = [
        'person_age', 'person_income', 'person_emp_exp', 'person_gender',
        'person_education', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"В CSV отсутствуют обязательные колонки: {', '.join(missing)}"
        )

    if require_target and 'loan_status' not in df.columns:
        raise ValueError(
            "Для расчёта ROC-AUC необходима колонка 'loan_status'"
        )


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Рассчитывает метрики качества.
    """
    from sklearn.metrics import roc_auc_score, confusion_matrix

    metrics = {}

    # ROC-AUC
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics['roc_auc'] = None

    # Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
    except Exception:
        metrics['confusion_matrix'] = None

    return metrics