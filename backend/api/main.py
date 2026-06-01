import os
import shutil
import sys
import tempfile
from pathlib import Path

# Добавляем backend в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import HealthResponse, ModelUploadResponse  # noqa: E402
from core.model import model_manager  # noqa: E402
from ml.config import PIPELINE_PATH  # noqa: E402


# Авто-загрузка модели при старте
print("Checking for model...")
if PIPELINE_PATH.exists():
    print(f"Found model: {PIPELINE_PATH}")
    success = model_manager.load_model(str(PIPELINE_PATH))
    if success:
        print(f"[OK] Model auto-loaded: {model_manager.model_type}")
    else:
        print("[ERROR] Failed to load model")
else:
    print(f"[WARNING] Model not found at: {PIPELINE_PATH}")
    print("   Upload via UI or place pipeline.pkl in backend/ml/artifacts/")


app = FastAPI(
    title="ML Сервис одобрения ипотеки",
    description="API для предсказания одобрения ипотечных заявок",
    version="1.0.0",
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка статических файлов фронтенда
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount(
        "/static", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css"
    )
    app.mount(
        "/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js"
    )
    print(f"[OK] Frontend static files mounted from: {FRONTEND_DIR}")
else:
    print(f"[WARNING] Frontend directory not found at: {FRONTEND_DIR}")


@app.get("/")
async def root():
    """Отдаем главную страницу"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ML Service API. Frontend not found."}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка статуса сервиса и модели"""
    status = model_manager.get_status()
    return HealthResponse(
        status="ok",
        model_loaded=status["loaded"],
        model_type=status["model_type"],
        version="1.0.0",
    )


@app.get("/model-status")
async def model_status():
    """Статус загрузки модели (без лишних полей)"""
    status = model_manager.get_status()
    return {
        "model_loaded": status["loaded"],
        "model_type": status["model_type"],
    }


@app.post("/upload-model", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)):
    """Загрузка модели (.pkl файл)"""
    if not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400, detail="Файл должен иметь расширение .pkl"
        )

    # Сохраняем временный файл
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Загружаем модель
        success = model_manager.load_model(temp_path)

        if success:
            return ModelUploadResponse(
                success=True,
                message="Модель успешно загружена",
                model_type=model_manager.model_type,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Не удалось загрузить модель. Проверьте файл.",
            )

    finally:
        # Очищаем временные файлы
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/predict")
async def predict(
    data: list = Body(..., description="Список записей с признаками клиента"),
):
    """
    Принимает одну или несколько записей с признаками клиента.
    Возвращает массив с признаками и loan_status (одобрено / отказ).
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Модель не загружена. Сначала загрузите модель через /upload-model",
            },
        )

    try:
        df = pd.DataFrame(data)

        # Делаем предсказание через pipeline (он сам обработает категориальные признаки)
        predictions, probabilities = model_manager.predict(df)

        results = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            row["loan_status"] = (
                "Approved" if int(predictions[i]) == 1 else "Rejected"
            )
            row["probability"] = round(float(probabilities[i][1]), 4)
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
async def predict_from_csv(file: UploadFile = None):
    """
    Принимает CSV-файл (сырой или предобработанный).
    Возвращает ROC-AUC (если есть loan_status) и полный датасет с predicted_loan_status.
    """
    if file is None:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Файл не передан. Загрузите CSV-файл.",
            },
        )

    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Модель не загружена. Сначала загрузите модель через /upload-model",
            },
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Файл должен быть в формате .csv",
            },
        )

    try:
        # Читаем CSV
        df = pd.read_csv(file.file)

        has_target = "loan_status" in df.columns
        y_true = df["loan_status"].values if has_target else None

        # Делаем предсказания через pipeline
        predictions, probabilities = model_manager.predict(df)

        # Вычисляем ROC-AUC если есть целевая переменная
        roc_auc = None
        if has_target and len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score

            roc_auc = round(float(roc_auc_score(y_true, probabilities[:, 1])), 4)

        results = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            row["predicted_loan_status"] = (
                "Approved" if int(predictions[i]) == 1 else "Rejected"
            )
            row["probability"] = round(float(probabilities[i][1]), 4)
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


def run():
    """Функция для запуска через скрипт"""
    import uvicorn

    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()