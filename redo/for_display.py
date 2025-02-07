import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
from threading import Thread
from ollama import Client  # Replace AsyncClient with Client for synchronous
import time

# Initialize the Ollama client
client = Client()

# Load the trained model
model = load_model("models/rowing_technique_model.h5")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Initialize the flag for controlling feedback requests
feedback_in_progress = False


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ab = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def end_of_stroke_detected(previous_knee_angle, current_knee_angle, previous_arm_angle, current_arm_angle, in_flexion, in_arm_extension):
    """Detects the end of a stroke by identifying the transition to extended legs and bent arms."""
    knee_extension_threshold = 160
    knee_flexion_threshold = 110
    arm_flexion_threshold = 60
    arm_extension_threshold = 150

    knee_extended = in_flexion and current_knee_angle > knee_extension_threshold
    arm_bent = in_arm_extension and current_arm_angle < arm_flexion_threshold

    if knee_extended and arm_bent:
        return True, False, False

    in_flexion = current_knee_angle < knee_flexion_threshold if not in_flexion else in_flexion
    in_arm_extension = current_arm_angle > arm_extension_threshold if not in_arm_extension else in_arm_extension

    return False, in_flexion, in_arm_extension

def generate_feedback_prompt(elbow_angle, knee_angle, trunk_angle, stroke_classification):
    """Generate a prompt for Ollama to provide feedback."""
    return f"""
    You are a rowing coach analyzing a rower. 
    Analyze the following rowing posture:
    - Elbow Angle: {elbow_angle:.2f} degrees
    - Knee Angle: {knee_angle:.2f} degrees
    - Trunk Angle: {trunk_angle:.2f} degrees

    The stroke was classified as '{stroke_classification}'.
    Provide specific feedback for the rower to improve their form if needed.
    Respond in around 6 words.
    Do not mention angles, only use things the rower can correct during their stroke.
    """

def get_ollama_feedback(prompt):
    """Synchronously fetch feedback from Ollama by running the asynchronous function."""
    feedback = ""
    try:
        response = client.chat(model="llama3", messages=[{"role": "user", "content": prompt}], stream=False)
        feedback = response['choices'][0]['message']['content']
    except Exception as e:
        feedback = f"Error: {e}"
    return feedback

def request_feedback(prompt):
    """Start a new thread to request Ollama feedback."""
    global feedback_in_progress, feedback_text
    if not feedback_in_progress:
        feedback_in_progress = True
        feedback_text = get_ollama_feedback(prompt)
        feedback_in_progress = False

def real_time_feedback():
    global feedback_in_progress
    cap = cv2.VideoCapture("BASIC/video/wyattGood.MOV")
    sequence = []
    previous_knee_angle = None
    previous_arm_angle = None
    in_flexion = True
    in_arm_extension = True
    feedback_text = "Processing..."
    stroke_classification = "Waiting for stroke..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 720)) #(640, 480))
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

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)
            trunk_angle = calculate_angle(hip, shoulder, knee)

            sequence.append([elbow_angle, knee_angle, trunk_angle])

            if previous_knee_angle is not None and previous_arm_angle is not None:
                stroke_ended, in_flexion, in_arm_extension = end_of_stroke_detected(
                    previous_knee_angle, knee_angle, previous_arm_angle, elbow_angle, in_flexion, in_arm_extension
                )
                if stroke_ended:
                    # Classify stroke
                    sequence_padded = pad_sequences([sequence], maxlen=50, padding="post", dtype="float32")
                    prediction = model.predict(sequence_padded)
                    stroke_classification = "Good" if prediction[0][0] >= 0.5 else "Bad"

                    # Generate feedback prompt for Ollama if stroke is bad
                    if stroke_classification == "Bad" and not feedback_in_progress:
                        prompt = generate_feedback_prompt(elbow_angle, knee_angle, trunk_angle, stroke_classification)
                        feedback_thread = Thread(target=request_feedback, args=(prompt,))
                        feedback_thread.start()

                    # Clear sequence for the next stroke
                    sequence = []

            previous_knee_angle = knee_angle
            previous_arm_angle = elbow_angle

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display feedback and classification on the video frame in the top left corner
        cv2.putText(frame, f"Stroke: {stroke_classification}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Feedback: {feedback_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if timer > 5:
            cv2.putText(frame, "embrace core keep shoulders back", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the video with landmarks and feedback
        cv2.imshow("Rowing Technique Feedback", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time feedback
real_time_feedback()
