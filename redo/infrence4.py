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

def end_of_stroke_detected(previous_knee_angle, current_knee_angle, previous_arm_angle, current_arm_angle, in_flexion, in_arm_extension):
    """
    Detects the end of a stroke by identifying the transition to extended legs and bent arms.
    """
    # Define thresholds for knee and arm positions for end and start of a stroke
    knee_extension_threshold = 160  # Legs flat
    knee_flexion_threshold = 110    # Legs bent
    arm_flexion_threshold = 60      # Arms bent
    arm_extension_threshold = 150   # Arms straight

    # Detect if the stroke has ended (legs flat, arms bent)
    knee_extended = in_flexion and current_knee_angle > knee_extension_threshold
    arm_bent = in_arm_extension and current_arm_angle < arm_flexion_threshold

    # If both conditions are met, we’ve reached the end of a stroke
    if knee_extended and arm_bent:
        return True, False, False  # End of stroke, reset flexion/extension states

    # Update flexion states based on current angles (preparing for next stroke)
    in_flexion = current_knee_angle < knee_flexion_threshold if not in_flexion else in_flexion
    in_arm_extension = current_arm_angle > arm_extension_threshold if not in_arm_extension else in_arm_extension

    return False, in_flexion, in_arm_extension

def real_time_feedback():
    cap = cv2.VideoCapture("BASIC/video/testStrokes/bad/stroke-bad.mp4-1.mp4")  # "BASIC/video/testStrokes/bad/stroke-bad.mp4-1.mp4"
    sequence = []
    previous_knee_angle = None
    previous_arm_angle = None
    in_flexion = True  # Start with legs in flexed (bent) position
    in_arm_extension = True  # Start with arms in extended (straight) position
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
                stroke_ended, in_flexion, in_arm_extension = end_of_stroke_detected(
                    previous_knee_angle, knee_angle, previous_arm_angle, elbow_angle, in_flexion, in_arm_extension
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
