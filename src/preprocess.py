import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Stratified train-test split and feature scaling
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
