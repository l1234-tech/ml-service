import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)

SAMPLE_RECORD = {
    "person_age": 30,
    "person_income": 50000,
    "person_emp_exp": 5,
    "person_gender": "male",
    "person_education": "Bachelor",
    "person_home_ownership": "RENT",
    "loan_amnt": 100000,
    "loan_intent": "EDUCATION",
    "loan_int_rate": 10.5,
    "loan_percent_income": 20,
    "cb_person_cred_hist_length": 5,
    "credit_score": 700,
    "previous_loan_defaults_on_file": "No",
}


def test_health_without_model():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False


def test_predict_without_model():
    response = client.post("/predict", json=[SAMPLE_RECORD])
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