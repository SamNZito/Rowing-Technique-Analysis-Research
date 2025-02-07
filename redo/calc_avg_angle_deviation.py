import pandas as pd
import numpy as np

# Load CSV file
file_path = "good_rowing_data_wyatt.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Define ideal angles for each phase (Catch & Finish)
ideal_angles = {
    "Catch": {"Elbow": 175, "Knee": 47.3, "Trunk": 24.5},
    "Finish": {"Elbow": 65, "Knee": 160, "Trunk": 26.3+90}
}

# Filter data by phase
catch_data = df[df["Phase"] == "Catch"]
finish_data = df[df["Phase"] == "Finish"]

# Function to calculate deviations
def compute_deviation(data, phase):
    elbow_dev = np.abs(data["Elbow Angle"] - ideal_angles[phase]["Elbow"])
    knee_dev = np.abs(data["Knee Angle"] - ideal_angles[phase]["Knee"])
    trunk_dev = np.abs(data["Hip Angle"] - ideal_angles[phase]["Trunk"])
    
    # Compute average deviation per frame
    frame_deviation = (elbow_dev + knee_dev + trunk_dev) / 3
    return frame_deviation

# Compute deviations for Catch and Finish
catch_deviation = compute_deviation(catch_data, "Catch")
finish_deviation = compute_deviation(finish_data, "Finish")



# Combine both phases
all_deviations = np.concatenate((catch_deviation, finish_deviation))

# Compute overall average posture deviation
avg_posture_deviation = np.mean(all_deviations)


# Compute deviations per joint
avg_elbow_dev = np.mean(np.concatenate((catch_deviation, finish_deviation)))
avg_knee_dev = np.mean(np.concatenate((catch_data["Knee Angle"] - ideal_angles["Catch"]["Knee"], 
                                       finish_data["Knee Angle"] - ideal_angles["Finish"]["Knee"])))
avg_trunk_dev = np.mean(np.concatenate((catch_data["Hip Angle"] - ideal_angles["Catch"]["Trunk"], 
                                        finish_data["Hip Angle"] - ideal_angles["Finish"]["Trunk"])))

print(f"Total Average Posture Deviation: {avg_posture_deviation:.2f}°")
print(f"Average Elbow Deviation: {avg_elbow_dev:.2f}°")
print(f"Average Knee Deviation: {avg_knee_dev:.2f}°")
print(f"Average Trunk Deviation: {avg_trunk_dev:.2f}°")


print(f"Average Posture Deviation: {avg_posture_deviation:.2f}°")
