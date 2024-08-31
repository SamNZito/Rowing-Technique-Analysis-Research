import cv2
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors

print("hello")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set desired frame size (width, height) that fits your screen
monitor = get_monitors()[0]
target_width = monitor.width  #800 Adjust this to your screen width
target_height = monitor.height  #600 Adjust this to your screen height

fScale = 0.8

# calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 200.0:
        angle = 360 - angle
    return angle

def draw_angle_with_arc(frame, p1, p2, p3, angle, color=(255, 255, 255)):
    p1 = tuple(np.multiply(p1, [frame.shape[1], frame.shape[0]]).astype(int))
    p2 = tuple(np.multiply(p2, [frame.shape[1], frame.shape[0]]).astype(int))
    p3 = tuple(np.multiply(p3, [frame.shape[1], frame.shape[0]]).astype(int))
    
    #Draw lines between points ---- can add back
    # cv2.line(frame, p1, p2, color, 2)
    # cv2.line(frame, p2, p3, color, 2)

    # Calculate the center of the angle (bending point)
    angle_center = p2
    
    # Draw the arc (half-circle)
    cv2.ellipse(frame, angle_center, (50, 50), 0, 0, -angle, color, 2)

    fcolor = (255, 255, 255)

    # Display the angle value
    cv2.putText(frame, str(int(angle)), angle_center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, fcolor, 2, cv2.LINE_AA)

def extract_key_points(landmarks):
    return {
        'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
        'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
        'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
        'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
        'nose': [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                 landmarks[mp_pose.PoseLandmark.NOSE.value].y],
        'neck': [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                 (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2]
    }


# Load a video
video_source = 'images/erg.webm'#'images/rowing2.mp4'  # Replace with 0 to use webcam
cap = cv2.VideoCapture(video_source)

paused = False

# Initialize the Pose model
with mp_pose.Pose(min_detection_confidence=0.5) as pose:
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to read the video")
                break

            # Convert the frame to RGB (MediaPipe expects RGB input)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform pose detection
            results = pose.process(frame_rgb)

            # Check if landmarks are detected
            if results.pose_landmarks:
                # Extract key points
                key_points = extract_key_points(results.pose_landmarks.landmark)
                
                # Calculate angles and draw arcs
                left_elbow_angle = calculate_angle(key_points['left_shoulder'], key_points['left_elbow'], key_points['left_wrist'])
                right_elbow_angle = calculate_angle(key_points['right_shoulder'], key_points['right_elbow'], key_points['right_wrist'])

                left_shoulder_angle = calculate_angle(key_points['left_hip'], key_points['left_shoulder'], key_points['left_elbow'])
                right_shoulder_angle = calculate_angle(key_points['right_hip'], key_points['right_shoulder'], key_points['right_elbow'])

                left_knee_angle = calculate_angle(key_points['left_hip'], key_points['left_knee'], key_points['left_ankle'])
                right_knee_angle = calculate_angle(key_points['right_hip'], key_points['right_knee'], key_points['right_ankle'])
                
                left_hip_angle = calculate_angle(key_points['left_knee'], key_points['left_hip'], key_points['left_shoulder'])
                right_hip_angle = calculate_angle(key_points['right_knee'], key_points['right_hip'], key_points['right_shoulder'])
                
                head_tilt_angle = calculate_angle(key_points['neck'], key_points['nose'], key_points['left_hip'])  # or use 'right_hip' for right side

                # Draw angles with arcs
                draw_angle_with_arc(frame, key_points['left_shoulder'], key_points['left_elbow'], key_points['left_wrist'], left_elbow_angle)
                draw_angle_with_arc(frame, key_points['right_shoulder'], key_points['right_elbow'], key_points['right_wrist'], right_elbow_angle)
                draw_angle_with_arc(frame, key_points['left_hip'], key_points['left_shoulder'], key_points['left_elbow'], left_shoulder_angle)
                draw_angle_with_arc(frame, key_points['right_hip'], key_points['right_shoulder'], key_points['right_elbow'], right_shoulder_angle)
                draw_angle_with_arc(frame, key_points['left_hip'], key_points['left_knee'], key_points['left_ankle'], left_knee_angle)
                draw_angle_with_arc(frame, key_points['right_hip'], key_points['right_knee'], key_points['right_ankle'], right_knee_angle)
                draw_angle_with_arc(frame, key_points['left_knee'], key_points['left_hip'], key_points['left_shoulder'], left_hip_angle)
                draw_angle_with_arc(frame, key_points['right_knee'], key_points['right_hip'], key_points['right_shoulder'], right_hip_angle)
                draw_angle_with_arc(frame, key_points['neck'], key_points['nose'], key_points['left_hip'], head_tilt_angle)
                
                color = (0, 0, 0)
                
                # Display calculated angles on the frame
                cv2.putText(frame, f'Left Elbow Angle: {int(left_elbow_angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Right Elbow Angle: {int(right_elbow_angle)}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Left Shoulder Angle: {int(left_shoulder_angle)}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Right Shoulder Angle: {int(right_shoulder_angle)}', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Left Knee Angle: {int(left_knee_angle)}', (50, 170), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Right Knee Angle: {int(right_knee_angle)}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Left Hip Angle: {int(left_hip_angle)}', (50, 230), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Right Hip Angle: {int(right_hip_angle)}', (50, 260), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
                cv2.putText(frame, f'Head Tilt Angle: {int(head_tilt_angle)}', (50, 290), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)

                # Draw the landmarks and connections on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Resize the frame after drawing everything
            frame_resized = cv2.resize(frame, (target_width, target_height))
            
            # Display the frame with landmarks
            cv2.imshow('Pose Estimation', frame_resized)

        # Check for key presses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar pressed
            paused = not paused  # Toggle pause state

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
