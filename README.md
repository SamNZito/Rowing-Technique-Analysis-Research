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
- rowing_technique_model.h5     # Pretrained TensorFlow model
- redo/                 # Trained AI models
- inference7.py     # Runs the Actual working version of AI system
- scripts:                # Data processing and AI training scripts
- preprocess.py       # Extracts joint angles from raw data
- train_model.py      # Trains the AI model on labeled data
- evaluate_model.py   # Computes accuracy and posture deviation
- README.md               # Project documentation

## File Structure
```

─Rowing-Technique-Analysis-Research
    │   .gitignore
    │   angle.py
    │   bad_rowing_data.csv
    │   bad_rowing_data_wyatt.csv
    │   good_rowing_data_wyatt.csv
    │   how-I-did-venv.txt
    │   pose-estimation.py
    │   README.md
    │   requirements.txt
    │   testing.csv
    │   
    ├───BASIC
    │   └───json_data
    │       └───dataset
    │           ├───bad
    │           │       bad_stroke-01.json
    │           │       ...
    │           │
    │           ├───good
    │           │       good_stroke-1.json
    │           │       ...
    │           │
    │           └───testData
    │               ├───bad
    │               │       test_stroke-bad.mp4-1.json
    │               │       ...
    │               │
    │               └───good
    │                       test_stroke-good.mp4-1.json
    │                       ...
    │
    ├───images
    │       erg.jpg
    │       ...
    │
    ├───models
    │       rowing_technique_model.h5
    │
    └───redo
        │   calc_avg_angle_deviation.py
        │   capture_stroke.py
        │   data_loader.py
        │   draw_body_landmarks.py
        │   evaluate.py
        │   extract_key_points.py
        │   for_display.py
        │   inference_v1.1.py
        │   infrence.py
        │   infrence2.py
        │   infrence3.py
        │   infrence4.py
        │   infrence5.py
        │   infrence6.py
        │   infrence7.py
        │   infrence8.py
        │   model.py
        │   tempCodeRunnerFile.py
        │   train.py
        │
        └───important_documents_research_paper
                IMG_3487.jpg
                IMG_3488.jpg
                IMG_3489.jpg
                Methodology.drawio.png
                model.png
                Screenshot 2024-11-10 173456.png
                Screenshot 2024-11-10 173722.png
                Screenshot 2024-11-14 193255.png

```

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