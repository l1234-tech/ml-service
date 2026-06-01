import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt


def get_models():
    """Возвращает словарь моделей с пайплайнами."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=42, n_estimators=100)),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("scaler", StandardScaler()),
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
    return {"auc": auc, "report": report, "y_pred": y_pred, "y_proba": y_proba}


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