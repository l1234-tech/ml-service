from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_model_status_without_model():
    response = client.get("/model-status")
    assert response.status_code == 200
    assert response.json() == {"loaded": False}


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
    data = response.json()
    assert data["detail"]["status"] == "error"
    assert ".pkl" in data["detail"]["message"]


def test_upload_model_invalid_content():
    """Попытка загрузить невалидный .pkl файл."""
    response = client.post(
        "/upload-model",
        files={"file": ("model.pkl", b"not a real pickle file", "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["status"] == "error"


def test_clear_model():
    response = client.delete("/clear-model")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_model_status_after_clear():
    response = client.get("/model-status")
    assert response.status_code == 200
    assert response.json() == {"loaded": False}


def test_predict_empty_list():
    """Пустой массив записей."""
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
    assert ".csv" in data["detail"]["message"]