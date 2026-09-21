from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):
    """
    Print evaluation metrics for fraud detection
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    return confusion_matrix(y_test, y_pred)
