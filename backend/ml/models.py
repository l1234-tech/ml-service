import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from .config import RANDOM_STATE
from .preprocessing import create_preprocessing_pipeline, create_feature_selector


def create_model_pipeline(model) -> Pipeline:
    preprocessor = create_preprocessing_pipeline()
    selector = create_feature_selector()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", model),
        ]
    )


def get_models() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }
    return {name: create_model_pipeline(m) for name, m in models.items()}


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "auc": auc,
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
        "y_proba": y_proba,
    }


def plot_roc_curves(results: dict, output_path: str) -> None:
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
