import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ml.config import (  # noqa: E402
    DATA_PATH,
    ARTIFACTS_DIR,
    PIPELINE_PATH,
    METRICS_PATH,
    SAMPLE_PAYLOAD_PATH,
    ROC_CURVES_PATH,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    IQR_COLS,
)
from ml.preprocessing import apply_iqr_capping  # noqa: E402
from ml.models import get_models, evaluate_model, plot_roc_curves  # noqa: E402
from ml.utils import save_object, save_json  # noqa: E402


def main():
    print("=" * 60)
    print("ML Pipeline Training")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n1. Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}")

    print("\n2. EDA")
    print("   df.info():")
    df.info()
    print("\n   df.describe():")
    print(df.describe())
    print("\n   df.head():")
    print(df.head())
    print("\n   Target distribution:")
    print(df[TARGET_COL].value_counts(normalize=True))

    print("\n3. Missing values:")
    print(df.isnull().sum())

    print("\n4. IQR Capping")
    df = apply_iqr_capping(df, IQR_COLS)

    print("\n5. Train/Test split")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n6. Training models")
    models = get_models()
    results = {}
    for name, pipeline in models.items():
        print(f"   Training {name}...")
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        print(f"   {name} ROC-AUC: {results[name]['auc']:.4f}")

    print("\n7. Model comparison")
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_pipeline = models[best_name]
    print(f"\n   {'=' * 40}")
    print(f"   Best model: {best_name}")
    print(f"   ROC-AUC: {results[best_name]['auc']:.4f}")
    print(f"   {'=' * 40}")

    for name, res in results.items():
        print(f"   {name}: ROC-AUC = {res['auc']:.4f}")

    print("\n8. Saving artifacts")
    save_object(best_pipeline, PIPELINE_PATH)
    print(f"   Pipeline saved to {PIPELINE_PATH}")

    metrics_report = {
        name: {
            "roc_auc": round(res["auc"], 4),
            "accuracy": round(res["report"]["accuracy"], 4),
            "precision_0": round(res["report"]["0"]["precision"], 4),
            "recall_0": round(res["report"]["0"]["recall"], 4),
            "f1_0": round(res["report"]["0"]["f1-score"], 4),
            "precision_1": round(res["report"]["1"]["precision"], 4),
            "recall_1": round(res["report"]["1"]["recall"], 4),
            "f1_1": round(res["report"]["1"]["f1-score"], 4),
        }
        for name, res in results.items()
    }
    metrics_report["best_model"] = best_name
    save_json(metrics_report, METRICS_PATH)
    print(f"   Metrics saved to {METRICS_PATH}")

    sample_data = X_test.head(5).to_dict(orient="records")
    save_json(sample_data, SAMPLE_PAYLOAD_PATH)
    print(f"   Sample payload saved to {SAMPLE_PAYLOAD_PATH}")

    plot_roc_curves(results, ROC_CURVES_PATH)
    print(f"   ROC curves saved to {ROC_CURVES_PATH}")

    print("\n9. Testing loaded pipeline")
    try:
        from ml.utils import load_object

        loaded_pipeline = load_object(PIPELINE_PATH)
        test_sample = pd.DataFrame(sample_data)
        test_pred = loaded_pipeline.predict(test_sample)
        test_proba = loaded_pipeline.predict_proba(test_sample)
        print("   Loaded pipeline test passed!")
        print(f"   Predictions: {test_pred}")
        print(f"   Probabilities: {test_proba[:, 1]}")
    except Exception as e:
        print(f"   ERROR loading pipeline: {e}")
        raise

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()