from tensorflow.keras.models import load_model
from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Paths
dataset_path = "BASIC/json_data/dataset/testData"
model_path = "models/rowing_technique_model.h5"

# Load data
X, y = load_data(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = load_model(model_path)

# Evaluate model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 1. Generate predictions (probabilities)
y_pred_probs = model.predict(X_test)

# 2. Convert probabilities to discrete class labels (0 or 1)
#    If your task is binary classification, you can do the following:
y_pred_classes = (y_pred_probs > 0.5).astype("int32")

# 3. Compute precision, recall, and f1
precision = precision_score(y_test, y_pred_classes, average='binary')
recall = recall_score(y_test, y_pred_classes, average='binary')
f1 = f1_score(y_test, y_pred_classes, average='binary')

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# from tensorflow.keras.models import load_model
# from data_loader import load_data
# from sklearn.model_selection import train_test_split

# # Paths
# dataset_path = "BASIC/json_data/dataset/testData"
# model_path = "models/rowing_technique_model.h5"

# # Load data
# X, y = load_data(dataset_path)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Load model
# model = load_model(model_path)

# # Evaluate model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")
