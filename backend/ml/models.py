def evaluate_model(pipeline, X_test, y_test):
    """Оценка модели: ROC-AUC и classification report."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {"auc": auc, "report": report, "y_pred": y_pred, "y_proba": y_proba, "y_test": y_test}