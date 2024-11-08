import os
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(dataset_path, sequence_length=50):
    sequences = []
    labels = []
    
    for label_type in ["good", "bad"]:
        label_dir = os.path.join(dataset_path, label_type)
        label = 1 if label_type == "good" else 0
        
        for file in os.listdir(label_dir):
            if file.endswith(".json"):
                file_path = os.path.join(label_dir, file)
                
                with open(file_path, "r") as f:
                    stroke_data = json.load(f)
                
                # Extract angles and convert to numpy array
                sequence = np.array([[frame["elbow_angle"], frame["knee_angle"], frame["trunk_angle"]] for frame in stroke_data])
                
                # Append the sequence and label
                sequences.append(sequence)
                labels.append(label)
    
    # Pad sequences for consistent length
    X = pad_sequences(sequences, maxlen=sequence_length, padding="post", dtype="float32")
    y = np.array(labels)
    
    return X, y
