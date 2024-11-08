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
    """Calculate the angle between three points."""
    ab = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def end_of_stroke_detected(previous_knee_angle, current_knee_angle, previous_arm_angle, current_arm_angle, in_flexion, in_arm_flexion):
    """
    Detects the end of a stroke by identifying a complete knee and arm angle cycle from flexion to extension and back.
    """
    # Define thresholds for knee and arm flexion and extension
    knee_flexion_threshold = 110  # Knee flexion threshold
    knee_extension_threshold = 160  # Knee extension threshold
    arm_flexion_threshold = 60     # Arm flexion threshold
    arm_extension_threshold = 150  # Arm extension threshold

    # Check if knee and arm have moved from flexion to extension or vice versa
    knee_transition = in_flexion and current_knee_angle > knee_extension_threshold
    arm_transition = in_arm_flexion and current_arm_angle > arm_extension_threshold

    # If both knee and arm have transitioned from flexion to extension, mark end of stroke
    if knee_transition and arm_transition:
        return True, False, False  # End of stroke, reset both flexion states

    # Update flexion states based on current angles
    in_flexion = current_knee_angle < knee_flexion_threshold if not in_flexion else in_flexion
    in_arm_flexion = current_arm_angle < arm_flexion_threshold if not in_arm_flexion else in_arm_flexion

    return False, in_flexion, in_arm_flexion

def real_time_feedback():
    cap = cv2.VideoCapture("BASIC/video/wyattGood.MOV")  # Capture from live camera feed
    sequence = []
    previous_knee_angle = None
    previous_arm_angle = None
    in_flexion = True  # Start with knee in flexed position
    in_arm_flexion = True  # Start with arm in flexed position
    feedback = "Waiting for stroke..."

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
            elbow_angle = calculate_angle(shoulder, elbow, wrist)  # Arm angle
            knee_angle = calculate_angle(hip, knee, ankle)
            trunk_angle = calculate_angle(hip, shoulder, knee)

            # Append angles to sequence
            sequence.append([elbow_angle, knee_angle, trunk_angle])

            # Detect end of stroke
            if previous_knee_angle is not None and previous_arm_angle is not None:
                stroke_ended, in_flexion, in_arm_flexion = end_of_stroke_detected(
                    previous_knee_angle, knee_angle, previous_arm_angle, elbow_angle, in_flexion, in_arm_flexion
                )
                if stroke_ended:
                    # Pad sequence and classify the stroke
                    sequence_padded = pad_sequences([sequence], maxlen=50, padding="post", dtype="float32")
                    prediction = model.predict(sequence_padded)
                    feedback = "Good" if prediction[0][0] >= 0.5 else "Bad"

                    # Clear sequence for the next stroke
                    sequence = []

            # Update previous angles
            previous_knee_angle = knee_angle
            previous_arm_angle = elbow_angle

        # Display feedback on the video feed
        cv2.putText(frame, f"Feedback: {feedback}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Rowing Technique Feedback", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time feedback function
real_time_feedback()
