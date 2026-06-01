from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_PATH = Path(r"datasets/loan_data.csv")

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics_report.json"
SAMPLE_PAYLOAD_PATH = ARTIFACTS_DIR / "sample_payload.json"
ROC_CURVES_PATH = ARTIFACTS_DIR / "roc_curves.png"
SAMPLE_CSV_PATH = ARTIFACTS_DIR / "sample_data.csv"
SAMPLE_CSV_WITH_TARGET_PATH = ARTIFACTS_DIR / "sample_data_with_target.csv"

# Пути для всех обученных моделей
ALL_MODELS_DIR = ARTIFACTS_DIR / "models"
LOGISTIC_REGRESSION_PATH = ALL_MODELS_DIR / "logistic_regression.pkl"
RANDOM_FOREST_PATH = ALL_MODELS_DIR / "random_forest.pkl"
GRADIENT_BOOSTING_PATH = ALL_MODELS_DIR / "gradient_boosting.pkl"

CATEGORICAL_COLS = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

TARGET_COL = "loan_status"

IQR_COLS = ["person_income", "loan_amnt", "person_age"]