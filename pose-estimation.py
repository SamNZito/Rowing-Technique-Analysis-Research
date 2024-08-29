# TO DO
# collect/process an image
# pose estimation or skeletoning image
# collect and process video data
# pose estimation or skeletoning people in videos
# feature extraction - extract key points 
# model development
# real time feed back system
# testing


import cv2
import mediapipe as mp
import numpy as np

print("hello")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set desired frame size (width, height) that fits your screen
target_width = 800  # Adjust this to your screen width
target_height = 600  # Adjust this to your screen height


# calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

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
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    }
    
# load a video
video_source = 'images/rowing2.mp4' # replace with 0 to use webcam
cap = cv2.VideoCapture(video_source)


# Initialize the Pose model
with mp_pose.Pose(min_detection_confidence=0.5) as pose:
    while cap.isOpened():
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
            # # Draw the landmarks and connections on the frame
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
             # Calculate angles
            left_elbow_angle = calculate_angle(key_points['left_shoulder'], key_points['left_elbow'], key_points['left_wrist'])
            right_elbow_angle = calculate_angle(key_points['right_shoulder'], key_points['right_elbow'], key_points['right_wrist'])

            left_shoulder_angle = calculate_angle(key_points['left_hip'], key_points['left_shoulder'], key_points['left_elbow'])
            right_shoulder_angle = calculate_angle(key_points['right_hip'], key_points['right_shoulder'], key_points['right_elbow'])

            color = (0, 0, 0)
            fScale = 0.8
            
            # Display calculated angles on the frame
            cv2.putText(frame, f'Left Elbow Angle: {int(left_elbow_angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
            cv2.putText(frame, f'Right Elbow Angle: {int(right_elbow_angle)}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
            cv2.putText(frame, f'Left Shoulder Angle: {int(left_shoulder_angle)}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)
            cv2.putText(frame, f'Right Shoulder Angle: {int(right_shoulder_angle)}', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, fScale, (color), 2)

            # Draw the landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Resize the frame to fit the screen
        frame_resized = cv2.resize(frame, (target_width, target_height))
        
        # Display the frame with landmarks
        cv2.imshow('Pose Estimation', frame_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# images --------------------
# # Load an image
# image = cv2.imread('images/erg5.jpg')


# # Check if the image is loaded correctly
# if image is None:
#     print("Error: Could not load image.")
#     exit()

# # Convert the image to RGB (MediaPipe expects RGB input)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Initialize the Pose model
# with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
#     # Perform pose detection
#     results = pose.process(image_rgb)

#     # Check if landmarks are detected
#     if results.pose_landmarks:
#         # Draw the landmarks and connections on the image
#         mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     # Convert back to BGR for OpenCV display
#     image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#     # Display the image with landmarks
#     cv2.imshow('Pose Estimation', image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()