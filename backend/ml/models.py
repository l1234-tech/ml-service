import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, roc_auc_score, roc_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from ml.config import CATEGORICAL_COLS, NUMERIC_COLS  # noqa: E402


def _build_preprocessor():
    """Строит ColumnTransformer: числовые -> StandardScaler, категориальные -> OneHotEncoder."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def get_models():
    """Возвращает словарь моделей с пайплайнами (предобработка + классификатор)."""
    preprocessor = _build_preprocessor()

    return {
        "LogisticRegression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clf", RandomForestClassifier(random_state=42, n_estimators=100)),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clf", GradientBoostingClassifier(random_state=42, n_estimators=100)),
            ]
        ),
    }


def evaluate_model(pipeline, X_test, y_test):
    """Оценка модели: ROC-AUC и classification report."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "auc": auc,
        "report": report,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_test": y_test,
    }


def plot_roc_curves(results, save_path):
    """Построение и сохранение ROC-кривых для всех моделей."""
    plt.figure(figsize=(10, 8))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
