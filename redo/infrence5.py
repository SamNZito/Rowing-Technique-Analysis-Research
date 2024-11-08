import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
import asyncio
from ollama import AsyncClient

# Load the trained model
model = load_model("models/rowing_technique_model.h5")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# Initialize Ollama client
client = AsyncClient()

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

async def get_ollama_feedback(prompt):
    """Get feedback from Ollama based on the rowing posture prompt."""
    async for part in await client.chat(
        model="llama3", messages=[{"role": "user", "content": prompt}], stream=True
    ):
        feedback = part["message"]["content"].strip()
        if feedback:
            print(f"Ollama Feedback: {feedback}")

def generate_feedback_prompt(elbow_angle, knee_angle, trunk_angle, stroke_classification):
    """Generate a prompt for Ollama to provide feedback."""
    return f"""
    You are a rowing coach/instructor, you are currently analyzing a rower on an erg machine. 
    Analyze the following rowing posture:
    - Elbow Angle: {elbow_angle:.2f} degrees
    - Knee Angle: {knee_angle:.2f} degrees
    - Trunk Angle: {trunk_angle:.2f} degrees

    The stroke was classified as '{stroke_classification}'.
    Provide specific feedback for the rower to improve their form if needed.
    Respond in around 6 words.
    Just tell rower what to fix.
    Don't mention angles but word it in a way they can fix their mistakes without having to look at themselves.
    Respond concisely with one or two actionable suggestions.
    """

def real_time_feedback():
    cap = cv2.VideoCapture("BASIC/video/wyattGood.MOV")  # "BASIC/video/testStrokes/bad/stroke-bad.mp4-1.mp4"
    sequence = []
    previous_knee_angle = None
    previous_arm_angle = None
    in_flexion = True  # Start with knee in flexed position
    in_arm_extension = True  # Start with arm in extended position
    feedback_text = "Waiting for stroke..."

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

            # Detect end of stroke
            if previous_knee_angle is not None and previous_arm_angle is not None:
                stroke_ended, in_flexion, in_arm_extension = end_of_stroke_detected(
                    previous_knee_angle, knee_angle, previous_arm_angle, elbow_angle, in_flexion, in_arm_extension
                )
                if stroke_ended:
                    # Pad sequence and classify the stroke
                    sequence_padded = pad_sequences([sequence], maxlen=50, padding="post", dtype="float32")
                    prediction = model.predict(sequence_padded)
                    stroke_classification = "Good" if prediction[0][0] >= 0.5 else "Bad"
                    feedback_text = stroke_classification

                    # Generate a prompt for Ollama feedback based on the angles and classification
                    prompt = generate_feedback_prompt(elbow_angle, knee_angle, trunk_angle, stroke_classification)
                    
                    # Run Ollama feedback asynchronously
                    asyncio.run(get_ollama_feedback(prompt))

                    # Clear sequence for the next stroke
                    sequence = []

            # Update previous angles
            previous_knee_angle = knee_angle
            previous_arm_angle = elbow_angle

        # Display feedback text on the video feed for debugging
        print(f"Model Feedback: {feedback_text}")

        # Show the frame
        cv2.imshow("Rowing Technique Feedback", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time feedback function
real_time_feedback()
