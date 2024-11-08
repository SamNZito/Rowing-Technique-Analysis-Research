import os
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import build_model

# Paths
dataset_path = "BASIC/json_data/dataset"
model_path = "models/rowing_technique_model.h5"

# Load data
X, y = load_data(dataset_path)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
input_shape = (X_train.shape[1], X_train.shape[2])  # Sequence length, features per frame
model = build_model(input_shape)

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=4)

# Save model
os.makedirs("models", exist_ok=True)
model.save(model_path)
print(f"Model saved to {model_path}")
