import joblib
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path="results/final_model.pkl"):
    joblib.dump(model, path)


def save_scaler(scaler, path="results/scaler.pkl"):
    joblib.dump(scaler, path)
