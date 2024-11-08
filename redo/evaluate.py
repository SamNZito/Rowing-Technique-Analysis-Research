from tensorflow.keras.models import load_model
from data_loader import load_data
from sklearn.model_selection import train_test_split

# Paths
dataset_path = "BASIC/json_data/dataset/testData"
model_path = "models/rowing_technique_model.h5"

# Load data
X, y = load_data(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = load_model(model_path)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
