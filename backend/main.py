import io
import joblib
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sklearn.metrics import roc_auc_score

# ─── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Service — предсказание одобрения кредитов")

# ─── Global state ──────────────────────────────────────────────────────────
model = None
model_loaded = False
is_pipeline = False

# ─── Feature order for preprocessed data ───────────────────────────────────
FEATURE_ORDER = [
    "person_age", "person_income", "person_emp_exp", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score",
    "person_gender_female", "person_gender_male",
    "person_education_Associate", "person_education_Bachelor",
    "person_education_Doctorate", "person_education_High School",
    "person_education_Master",
    "person_home_ownership_MORTGAGE", "person_home_ownership_OTHER",
    "person_home_ownership_OWN", "person_home_ownership_RENT",
    "loan_intent_DEBTCONSOLIDATION", "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL",
    "loan_intent_PERSONAL", "loan_intent_VENTURE",
    "previous_loan_defaults_on_file_No", "previous_loan_defaults_on_file_Yes",
]

# ─── Raw data detection ────────────────────────────────────────────────────
RAW_COLUMNS = {
    "person_gender", "person_education", "person_home_ownership",
    "loan_intent", "previous_loan_defaults_on_file",
}


def is_raw_data(df: pd.DataFrame) -> bool:
    """Определяет, сырые данные или уже предобработанные (one-hot)."""
    return bool(RAW_COLUMNS & set(df.columns))


# ─── Pydantic models ───────────────────────────────────────────────────────
class PredictItem(BaseModel):
    person_age: float = Field(..., ge=18, le=100, description="Возраст от 18 до 100")
    person_income: float = Field(..., gt=0, description="Доход должен быть > 0")
    person_emp_exp: float = Field(..., ge=0, le=70)
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0, le=100)
    loan_percent_income: float = Field(..., ge=0, le=500)
    cb_person_cred_hist_length: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)


# ─── Global exception handler ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"Внутренняя ошибка сервера: {str(exc)}",
            "endpoint": str(request.url),
        },
    )


# ─── Helper: check model loaded ────────────────────────────────────────────
def require_model():
    if not model_loaded:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Модель не загружена. Сначала загрузите модель через /upload-model",
            },
        )


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/model-status")
async def model_status():
    """Проверка статуса модели"""
    return {"loaded": model_loaded}


@app.delete("/clear-model")
async def clear_model():
    """Сброс модели"""
    global model, model_loaded, is_pipeline
    model = None
    model_loaded = False
    is_pipeline = False
    return {"status": "success", "message": "Модель удалена"}


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Загрузка модели в формате .pkl. Принимает как Pipeline, так и просто модель."""
    global model, model_loaded, is_pipeline

    if not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Файл должен быть в формате .pkl"},
        )

    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Файл слишком большой (макс 100MB)"},
        )

    try:
        model = joblib.load(io.BytesIO(contents))
        model_loaded = True
        is_pipeline = hasattr(model, "named_steps")

        # Определяем имя классификатора
        if is_pipeline and "model" in model.named_steps:
            model_name = type(model.named_steps["model"]).__name__
        elif is_pipeline:
            model_name = type(model[-1]).__name__
        else:
            model_name = type(model).__name__

        return {
            "status": "success",
            "message": f"Модель {model_name} успешно загружена",
        }
    except Exception as e:
        model = None
        model_loaded = False
        is_pipeline = False
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Ошибка загрузки модели: {str(e)}",
            },
        )


@app.post("/predict")
async def predict(data: list[dict]):
    """
    Принимает массив записей с признаками клиента.
    Возвращает массив с признаками + loan_status (одобрено / отказ).
    """
    require_model()

    try:
        df = pd.DataFrame(data)

        if is_pipeline and is_raw_data(df):
            # Сырые данные — прогоняем через Pipeline (встроенная предобработка)
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]
        else:
            # Предобработанные данные — используем FEATURE_ORDER
            X = df[FEATURE_ORDER].fillna(0)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

        results = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            row["loan_status"] = "Approved" if predictions[i] == 1 else "Rejected"
            row["probability"] = round(float(probabilities[i]), 4)
            results.append(row)

        return {"predictions": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Ошибка предсказания: {str(e)}",
            },
        )


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Принимает CSV-файл (сырой или предобработанный).
    Возвращает ROC-AUC (если есть loan_status) и полный датасет с predicted_loan_status.
    """
    require_model()

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Файл должен быть в формате .csv",
            },
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        has_target = "loan_status" in df.columns
        y_true = df["loan_status"] if has_target else None

        if is_pipeline and is_raw_data(df):
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]
        else:
            feature_cols = [c for c in FEATURE_ORDER if c in df.columns]
            X = df[feature_cols].fillna(0)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

        # Вычисляем ROC-AUC если есть целевая переменная
        roc_auc = None
        if has_target and y_true.nunique() > 1:
            roc_auc = round(float(roc_auc_score(y_true, probabilities)), 4)

        results = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            row["predicted_loan_status"] = "Approved" if predictions[i] == 1 else "Rejected"
            row["probability"] = round(float(probabilities[i]), 4)
            results.append(row)

        response = {
            "total_rows": len(df),
            "predictions": results,
        }
        if roc_auc is not None:
            response["roc_auc"] = roc_auc

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Ошибка обработки CSV: {str(e)}",
            },
        )