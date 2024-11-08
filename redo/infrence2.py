import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp

# Load the trained model
model = load_model("models/rowing_technique_model.h5")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

def calculate_angle(a, b, c):
    # Calculate angle between three points
    ab = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)
            trunk_angle = calculate_angle(hip, shoulder, knee)

            # Append angles to sequence
            sequence.append([elbow_angle, knee_angle, trunk_angle])
    
    cap.release()
    
    # Pad the sequence to the required length
    sequence_padded = pad_sequences([sequence], maxlen=50, padding="post", dtype="float32")
    return sequence_padded

def classify_stroke(video_path):
    # Process video to get formatted sequence
    sequence = process_video(video_path)
    
    # Use model to predict
    prediction = model.predict(sequence)
    return f"Good, {prediction[0][0]}" if prediction[0][0] >= 0.5 else f"Bad, {prediction[0][0]}"

# Test the classification
video_path = "BASIC/video/testStrokes/good/stroke-good.mp4-2.mp4"
result = classify_stroke(video_path)
print(f"Stroke classification: {result}")
