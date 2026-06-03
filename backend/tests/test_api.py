from fastapi.testclient import TestClient

from api.main import app
from core.model import model_manager

client = TestClient(app)


def test_model_status_without_model():
    # Сбрасываем состояние модели, так как при старте приложения
    # модель могла автоматически загрузиться из artifacts/pipeline.pkl
    model_manager._is_loaded = False
    model_manager._pipeline = None
    model_manager._model_type = None

    response = client.get("/model-status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False


def test_predict_without_model():
    response = client.post("/predict", json=[{"person_age": 30}])
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_predict_from_csv_without_model():
    response = client.post("/predict-from-csv")
    assert response.status_code == 400


def test_upload_model_invalid_extension():
    response = client.post(
        "/upload-model",
        files={"file": ("model.txt", b"dummy content", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_model_invalid_content():
    """Попытка загрузить невалидный .pkl файл."""
    response = client.post(
        "/upload-model",
        files={
            "file": (
                "model.pkl",
                b"not a real pickle file",
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 400


def test_predict_empty_list():
    """Пустой массив записей (модель не загружена -> 400)."""
    response = client.post("/predict", json=[])
    assert response.status_code == 400


def test_predict_from_csv_no_file():
    """Запрос без файла."""
    response = client.post("/predict-from-csv")
    assert response.status_code == 400


def test_predict_from_csv_invalid_extension():
    """Файл не .csv."""
    response = client.post(
        "/predict-from-csv",
        files={"file": ("data.txt", b"a,b,c\n1,2,3", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
