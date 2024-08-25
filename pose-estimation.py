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

print("hello")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load an image
image = cv2.imread('images/erg5.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert the image to RGB (MediaPipe expects RGB input)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the Pose model
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    # Perform pose detection
    results = pose.process(image_rgb)

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Draw the landmarks and connections on the image
        mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert back to BGR for OpenCV display
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Display the image with landmarks
    cv2.imshow('Pose Estimation', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
