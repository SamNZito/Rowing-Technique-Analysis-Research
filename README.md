# Rowing Technique Analysis Using AI Feedback

## Overview
This repository contains the dataset, AI model, and evaluation scripts for the research project **"Rowing Technique Analysis Using AI Feedback for Performance Enhancement."** The project aims to compare AI-based rowing feedback with traditional coaching by analyzing key joint angles (elbow, knee, and trunk) during strokes.

## Features
- **Data Collection:** Preprocessed rowing stroke data, including labeled joint angles.
- **AI Model:** A TensorFlow-based binary classification model that predicts whether a stroke has "good" or "bad" form.
- **Evaluation Metrics:** Includes classification accuracy, precision, recall, and F1-score.
- **Posture Deviation Analysis:** Computes deviations from ideal rowing angles based on biomechanical research.
- **Reproducibility:** Scripts for training, testing, and evaluating the AI model.

## Important Files
- testing.csv                # Raw and processed rowing stroke data
- models                 # Trained AI models
- ├── rowing_technique_model.h5     # Pretrained TensorFlow model
- redo/                 # Trained AI models
- ├── inference7.py     # Runs the Actual working version of AI system
- scripts:                # Data processing and AI training scripts
- ├── preprocess.py       # Extracts joint angles from raw data
- ├── train_model.py      # Trains the AI model on labeled data
- ├── evaluate_model.py   # Computes accuracy and posture deviation
- README.md               # Project documentation


## Installation
### Prerequisites
Ensure you have **Python 3.9+** installed along with the required dependencies.

```sh
pip install -r requirements.txt
```

## Running the AI Model
### 1. Data Collection

To capture rowing strokes and extract body key points:
```sh
python capture_stroke.py
python extract_key_points.py
```
### 2. Preprocessing and Visualization

Process rowing footage and visualize detected landmarks:
```sh
python draw_body_landmarks.py
```
### 3. Running Inference

Run AI model inference to classify strokes:
```sh
python inference7.py
```
For testing different inference versions:
```sh
python inference2.py
python inference3.py
...
python inference8.py
```
### 4. Evaluating the Model

Compute accuracy, precision, recall, and posture deviation:
```sh
python evaluate.py
python calc_avg_angle_deviation.py
```
## Results

The AI system was tested against expert rowing coaches, with key findings including:

    AI correctly classified strokes with ~69% accuracy.
    Posture deviation improved by 2.89° with AI guidance and 4.32° with coach guidance.
    Rower feedback highlighted AI’s consistency but preferred human adaptability.

## Posture Deviation Calculation

To quantify how much a rower deviates from optimal technique, we computed the average posture deviation using elbow, knee, and trunk angles:
​​
## Reproducibility

All data, trained models, and evaluation scripts are provided for full reproducibility. Researchers can modify hyperparameters, train on new datasets, and analyze technique deviations.

If you use this dataset or code, please cite:

@misc{zito2024rowing,
  author = {Samuel Zito},
  title = {Rowing Technique Analysis Using AI Feedback for Performance Enhancement},
  year = {2024},
  url = {https://github.com/SamNZito/Rowing-Technique-Analysis-Research/}
}