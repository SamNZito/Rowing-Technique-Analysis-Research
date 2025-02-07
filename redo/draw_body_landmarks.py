import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks_on_video():
    cap = cv2.VideoCapture("BASIC/video/wyattGood.MOV")  # Replace with your video path or use 0 for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame color from BGR to RGB
        frame = cv2.resize(frame, (960, 720)) #(640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow("Erg Rowing Body Landmarks", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
draw_landmarks_on_video()
