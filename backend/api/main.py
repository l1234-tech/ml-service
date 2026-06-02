import shutil
import sys
import traceback
from pathlib import Path

# Добавляем backend в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import Body, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.schemas import HealthResponse, ModelUploadResponse  # noqa: E402
from core.model import model_manager  # noqa: E402
from ml.config import (
    PIPELINE_PATH,
    ROC_CURVES_PATH,
    SAMPLE_CSV_PATH,
    SAMPLE_CSV_WITH_TARGET_PATH,
)  # noqa: E402

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


@app.post("/train-default-model")
async def train_default_model():
    """Обучить модель по умолчанию"""
    try:
        # Импортируем и запускаем обучение
        from ml.train import main as train_main

        train_main()

        # Загружаем обученную модель
        success = model_manager.load_model(str(PIPELINE_PATH))
        if success:
            return {
                "success": True,
                "message": "Модель успешно обучена и загружена",
                "model_type": model_manager.model_type,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Модель обучена, но не удалось её загрузить",
            )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обучении модели: {str(e)}",
        )


@app.get("/download-model")
async def download_model():
    """Скачать файл модели (.pkl)"""
    if not PIPELINE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": "Модель не найдена. Сначала загрузите или обучите модель.",
            },
        )

    return FileResponse(
        path=str(PIPELINE_PATH),
        filename="pipeline.pkl",
        media_type="application/octet-stream",
    )


@app.get("/roc-curve")
async def get_roc_curve():
    """Получить изображение ROC-кривой"""
    if not ROC_CURVES_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": "ROC-кривая не найдена. Сначала обучите модель.",
            },
        )

    return FileResponse(
        path=str(ROC_CURVES_PATH),
        media_type="image/png",
    )


@app.get("/sample-csv")
async def get_sample_csv():
    """Скачать sample CSV файл для проверки предсказания"""
    path = SAMPLE_CSV_WITH_TARGET_PATH
    filename = "sample_data_with_target.csv"

    if not path.exists():
        path = SAMPLE_CSV_PATH
        filename = "sample_data.csv"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": "Sample CSV не найден. Сначала обучите модель.",
            },
        )

    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="text/csv",
    )


@app.post("/upload-model", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)):
    """Загрузка модели (.pkl файл)"""
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Файл должен иметь расширение .pkl")

    # Создаём директорию для артефактов, если её нет
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Сохраняем файл модели в постоянное место
        with open(str(PIPELINE_PATH), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Загружаем модель
        success = model_manager.load_model(str(PIPELINE_PATH))

        if success:
            # Используем имя файла пользователя как название модели
            user_model_name = file.filename.replace(".pkl", "")
            return ModelUploadResponse(
                success=True,
                message="Модель успешно загружена",
                model_type=user_model_name,
            )
        else:
            # Если не удалось загрузить — удаляем файл
            try:
                PIPELINE_PATH.unlink(missing_ok=True)
            except PermissionError:
                pass
            raise HTTPException(
                status_code=400,
                detail="Не удалось загрузить модель. Проверьте файл.",
            )

    except HTTPException:
        raise
    except Exception as e:
        # Если ошибка — удаляем файл
        try:
            PIPELINE_PATH.unlink(missing_ok=True)
        except PermissionError:
            pass
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка при загрузке модели: {str(e)}",
        )


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
            row["loan_status"] = "Approved" if int(predictions[i]) == 1 else "Rejected"
            row["probability"] = round(float(probabilities[i][1]), 4)
            results.append(row)

        return {"predictions": results}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] predict: {str(e)}")
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

        # Удаляем целевую переменную из признаков (модель обучалась без неё)
        X = df.drop(columns=["loan_status"]) if has_target else df

        # Делаем предсказания через pipeline
        predictions, probabilities = model_manager.predict(X)

        # Вычисляем ROC-AUC если есть ��елевая переменная
        roc_auc = None
        if has_target and len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score

            roc_auc = round(float(roc_auc_score(y_true, probabilities[:, 1])), 4)

        results = []
        for i in range(len(X)):
            row = X.iloc[i].to_dict()
            row["predicted_loan_status"] = "Approved" if int(predictions[i]) == 1 else "Rejected"
            row["probability"] = round(float(probabilities[i][1]), 4)
            results.append(row)

        response = {
            "total_rows": len(X),
            "predictions": results,
        }
        if roc_auc is not None:
            response["roc_auc"] = roc_auc

        return response

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] predict-from-csv: {str(e)}")
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
# end of file
