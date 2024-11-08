import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ab = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Define directories for good and bad strokes
# data_dirs = {
#     "good": "BASIC/video/goodStrokes",
#     "bad": "BASIC/video/badStrokes"
# }
data_dirs = {
    "test": "BASIC/video/testStrokes/bad"
}

# Output directory for JSON files
output_dir = "BASIC/json_data"
os.makedirs(output_dir, exist_ok=True)

# Process each video file in good and bad strokes directories
for label, dir_path in data_dirs.items():
    for video_file in os.listdir(dir_path):
        if video_file.endswith(".mp4"):  # Ensure only video files are processed
            video_path = os.path.join(dir_path, video_file)
            cap = cv2.VideoCapture(video_path)
            stroke_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    # Extract key points or angles
                    landmarks = results.pose_landmarks.landmark
                    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

                    # Calculate angles between joints
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    knee_angle = calculate_angle(hip, knee, ankle)
                    trunk_angle = calculate_angle(hip, shoulder, knee)

                    # Append angles or positions for this frame
                    stroke_data.append({
                        "elbow_angle": elbow_angle,
                        "knee_angle": knee_angle,
                        "trunk_angle": trunk_angle
                    })

            # Save stroke data to JSON file with a descriptive name
            json_filename = f"{label}_{os.path.splitext(video_file)[0]}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w") as f:
                json.dump(stroke_data, f)

            cap.release()
            print(f"Processed and saved {json_filename}.")

