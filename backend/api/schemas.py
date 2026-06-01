from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    person_age: float = Field(..., ge=18, le=100, description="Возраст (лет)")
    person_income: float = Field(..., ge=0, description="Доход ($)")
    person_emp_exp: float = Field(..., ge=0, le=60, description="Стаж работы (годы)")
    person_gender: str = Field(..., description="Пол (male/female)")
    person_education: str = Field(..., description="Образование")
    person_home_ownership: str = Field(..., description="Тип жилья")
    loan_amnt: float = Field(..., ge=0, description="Сумма кредита ($)")
    loan_intent: str = Field(..., description="Цель кредита")
    loan_int_rate: float = Field(..., ge=0, le=100, description="Процентная ставка (%)")
    loan_percent_income: float = Field(..., ge=0, le=1000, description="Процент дохода")
    cb_person_cred_hist_length: float = Field(..., ge=0, description="Кредитная история (годы)")
    credit_score: float = Field(..., ge=300, le=850, description="Кредитный рейтинг")
    previous_loan_defaults_on_file: str = Field(..., description="Были ли просрочки (Yes/No)")


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = отказ, 1 = одобрено")
    probability: float = Field(..., ge=0, le=1, description="Вероятность одобрения")
    status: str = Field(..., description="Approved или Rejected")


class CsvPredictResponse(BaseModel):
    predictions: List[int] = Field(..., description="Список предсказаний")
    probabilities: List[float] = Field(..., description="Список вероятностей")
    rows_processed: int = Field(..., description="Количество обработанных строк")
    roc_auc: Optional[float] = Field(None, description="ROC-AUC (если передан loan_status)")
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Матрица ошибок")
    approved_count: int = Field(..., description="Количество одобренных")
    rejected_count: int = Field(..., description="Количество отказов")
    avg_probability: float = Field(..., description="Средняя вероятность одобрения")


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok/error")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    model_type: Optional[str] = Field(None, description="Тип модели")
    version: str = Field("1.0.0", description="Версия API")


class ModelUploadResponse(BaseModel):
    success: bool = Field(..., description="Успешно ли загружена модель")
    message: str = Field(..., description="Сообщение")
    model_type: Optional[str] = Field(None, description="Тип загруженной модели")
