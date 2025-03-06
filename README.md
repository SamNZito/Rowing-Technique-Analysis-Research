# Rowing Technique Analysis Using AI Feedback

## Overview
This repository contains the dataset, AI model, and evaluation scripts for the research project **"Rowing Technique Analysis Using AI Feedback for Performance Enhancement."** The project aims to compare AI-based rowing feedback with traditional coaching by analyzing key joint angles (elbow, knee, and trunk) during strokes.

## Features
- **Data Collection:** Preprocessed rowing stroke data, including labeled joint angles.
- **AI Model:** A TensorFlow-based binary classification model that predicts whether a stroke has "good" or "bad" form.
- **Evaluation Metrics:** Includes classification accuracy, precision, recall, and F1-score.
- **Posture Deviation Analysis:** Computes deviations from ideal rowing angles based on biomechanical research.
- **Reproducibility:** Scripts for training, testing, and evaluating the AI model.

## Repository Structure
├── testing.csv                # Raw and processed rowing stroke data
├── models/                 # Trained AI models
│   ├── rowing_technique_model.h5     # Pretrained TensorFlow model
├── redo/                 # Trained AI models
│   ├── inference7.py     # Runs the Actual working version of AI system
├── scripts/                # Data processing and AI training scripts
│   ├── preprocess.py       # Extracts joint angles from raw data
│   ├── train_model.py      # Trains the AI model on labeled data
│   ├── evaluate_model.py   # Computes accuracy and posture deviation
├── results/                # Experiment results and logs
│   ├── evaluation_metrics.json # Accuracy, precision, recall, etc.
├── README.md               # Project documentation


## Installation
### Prerequisites
Ensure you have **Python 3.9+** installed along with the required dependencies.

```sh
pip install -r requirements.txt


Running the AI Model
1. Data Preprocessing

Run the following script to preprocess raw rowing stroke data into structured CSV files:

python scripts/preprocess.py

2. Training the Model

Train the AI model using:

python scripts/train_model.py

3. Evaluating the Model

Evaluate classification performance and compute posture deviation:

python scripts/evaluate_model.py

Results

The AI system was tested against expert rowing coaches, with key findings including:

    AI correctly classified strokes with ~69% accuracy.
    Posture deviation improved by 2.89° with AI guidance and 4.32° with coach guidance.
    Rower feedback highlighted AI’s consistency but preferred human adaptability.

Reproducibility

All data, trained models, and evaluation scripts are provided for full reproducibility. Researchers can modify hyperparameters, train on new datasets, and analyze technique deviations.
Citation

If you use this dataset or code, please cite:

@misc{zito2024rowing,
  author = {Samuel Zito},
  title = {Rowing Technique Analysis Using AI Feedback for Performance Enhancement},
  year = {2024},
  url = {https://github.com/your-repo-link}
}
