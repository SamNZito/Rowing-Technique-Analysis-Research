import cv2
import mediapipe as mp
import numpy as np

# Initialize Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

def calculate_angle(a, b, c):
    ab = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

# Open video
cap = cv2.VideoCapture("BASIC/video/wyattGood.MOV")
frame_count = 0
stroke_segments = []
current_stroke = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Calculate key angles (e.g., knee and hip)
        knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
        
        hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE])

        # Detect start of a new stroke when knee angle becomes small (catch phase)
        if knee_angle < 100:
            if current_stroke:
                stroke_segments.append(current_stroke)  # Save the completed stroke
            current_stroke = []  # Start a new stroke

        current_stroke.append(frame_count)  # Add current frame to the stroke sequence

    frame_count += 1

cap.release()

# Save stroke segments for later labeling
print("Detected stroke segments:", stroke_segments)
