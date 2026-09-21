from src.preprocess import load_data, split_and_scale
from src.train import train_model, save_model, save_scaler

# Load and preprocess data
X, y = load_data("data/raw/creditcard.csv")
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

# Train model
model = train_model(X_train, y_train)

# Save model and scaler
save_model(model)
save_scaler(scaler)

print("Model and scaler saved successfully.")
