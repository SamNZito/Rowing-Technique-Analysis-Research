import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Path to the model
model_path = "models/rowing_technique_model.h5"
model = load_model(model_path)

def load_single_stroke(file_path, sequence_length=50):
    with open(file_path, "r") as f:
        stroke_data = json.load(f)
    sequence = np.array([[frame["elbow_angle"], frame["knee_angle"], frame["trunk_angle"]] for frame in stroke_data])
    sequence_padded = pad_sequences([sequence], maxlen=sequence_length, padding="post", dtype="float32")
    return sequence_padded

def classify_stroke(file_path):
    sequence = load_single_stroke(file_path)
    prediction = model.predict(sequence)
    return "Good" if prediction[0][0] >= 0.5 else "Bad"

# Example usage
stroke_file = "path_to_new_stroke.json"
result = classify_stroke(stroke_file)
print(f"Stroke classification: {result}")
