import os
import sys
from pathlib import Path
import tempfile
import shutil

# Добавляем backend в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

from api.schemas import (
    PredictRequest, 
    PredictResponse, 
    CsvPredictResponse,
    HealthResponse,
    ModelUploadResponse
)
from core.model import model_manager
from core.preprocessing import (
    create_dataframe_from_raw, 
    validate_csv_columns,
    calculate_metrics
)
from ml.config import PIPELINE_PATH


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
    version="1.0.0"
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка статических файлов фронтенда
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
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
        version="1.0.0"
    )


@app.post("/upload-model", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)):
    """Загрузка модели (.pkl файл)"""
    if not file.filename.endswith('.pkl'):
        raise HTTPException(
            status_code=400, 
            detail="Файл должен иметь расширение .pkl"
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
                model_type=model_manager.model_type
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Не удалось загрузить модель. Проверьте файл."
            )
    
    finally:
        # Очищаем временные файлы
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Одиночное предсказание"""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="Модель не загружена. Сначала загрузите .pkl файл через /upload-model"
        )
    
    try:
        # Преобразуем запрос в DataFrame
        data = request.model_dump()
        df = create_dataframe_from_raw(data)
        
        # Делаем предсказание
        predictions, probabilities = model_manager.predict(df)
        
        prediction = int(predictions[0])
        probability = float(probabilities[0][1])  # Вероятность класса 1
        
        return PredictResponse(
            prediction=prediction,
            probability=probability,
            status="Approved" if prediction == 1 else "Rejected"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


@app.post("/predict-csv", response_model=CsvPredictResponse)
async def predict_csv(
    file: UploadFile = File(...),
    has_target: bool = False
):
    """Предсказание по CSV файлу"""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="Модель не загружена. Сначала загрузите .pkl файл через /upload-model"
        )
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен иметь расширение .csv"
        )
    
    try:
        # Читаем CSV
        df = pd.read_csv(file.file)
        
        # Валидируем колонки
        validate_csv_columns(df, require_target=has_target)
        
        # Сохраняем target если есть
        y_true = None
        if has_target and 'loan_status' in df.columns:
            y_true = df['loan_status'].values
            df = df.drop('loan_status', axis=1)
        
        # Делаем предсказания
        predictions, probabilities = model_manager.predict(df)
        
        # Рассчитываем метрики
        roc_auc = None
        confusion_matrix = None
        if y_true is not None:
            metrics = calculate_metrics(y_true, predictions, probabilities[:, 1])
            roc_auc = metrics.get('roc_auc')
            confusion_matrix = metrics.get('confusion_matrix')
        
        # Статистика
        approved_count = int(np.sum(predictions == 1))
        rejected_count = int(np.sum(predictions == 0))
        avg_probability = float(np.mean(probabilities[:, 1]))
        
        return CsvPredictResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities[:, 1].tolist(),
            rows_processed=len(df),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix,
            approved_count=approved_count,
            rejected_count=rejected_count,
            avg_probability=avg_probability
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки CSV: {str(e)}")


def run():
    """Функция для запуска через скрипт"""
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
