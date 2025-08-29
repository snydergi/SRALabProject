import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.animation as animation

# Set plotting parameters
plt.rcParams['svg.fonttype'] = 'none'

# Load the data
# DATA ORDER:
# p_jt_pos1, p_jt_pos2, p_jt_pos3, p_jt_pos4, p_jt_vel1, p_jt_vel2, p_jt_vel3, p_jt_vel4, p_SIF, p_bp_pos, p_bp_vel,
# t_jt_pos1, t_jt_pos2, t_jt_pos3, t_jt_pos4, t_jt_vel1, t_jt_vel2, t_jt_vel3, t_jt_vel4
test_1 = pd.read_csv('../data/test_set_1.csv', skiprows=0).values
test_2 = pd.read_csv('../data/test_set_2.csv', skiprows=0).values
test_3 = pd.read_csv('../data/test_set_3.csv', skiprows=0).values

def create_dataset(dataset, lookback, step=1):
    """Transform time series data into a prediction dataset.
    
    Args:
        dataset (np.ndarray): An array of time series data. Shape [timesteps, features]
        lookback (int): Size of window for prediction
        step (int, optional): Step between consecutive windows. Defaults to 1.

    Returns:
        tuple:
            - X (torch.Tensor): Feature tensor, shape [samples, lookback, 8]
            - y (torch.Tensor): Target tensor, shape [samples, lookback, 1]

    Notes:
        - Change second feature index to select feature joint/joints
        - Change second target index to select target joint/joints
        - This version has a 25 step offset between windows to increase future prediction time.
    """
    X, y = [], []
    for i in range(0, len(dataset)-lookback-25, step):
        feature = dataset[i:i+lookback, :8]  # Feature is patient data, if input size 8
        target = dataset[i+25:i+lookback+25, predicted_target]  # Target is therapist data
        target = target.reshape(-1, 1)
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lookback, step = 50, 5
patient_number = 1 # Set based on which patient data is being tested
test = test_1 if patient_number == 1 else test_2 if patient_number == 2 else test_3
predicted_target = -8 # Index based on data order above
pIndex = 2 # Index of patient joint to plot against therapist joint (0-3 for patient joints 1-4)
X_test, y_test = create_dataset(test, lookback=lookback)

# Model definition
class JointModel(nn.Module):
    """A neural network model combining LSTM and linear layers for sequence prediction.
    
    This model processes sequential input data using an LSTM layer followed by a linear
    transformation layer to produce predictions for each timestep.
    
    Architecture:
        - LSTM layer: 1 layer, 50 hidden units, processes sequences with 8 input features
        - Linear layer: Maps from 50-dimensional LSTM output to 1-dimensional prediction
    """
    def __init__(self):
        """Initialize JointModel."""
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, 8]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, 1]
        """
        x, _ = self.lstm(x)
        return self.linear(x)

# Load model
model = JointModel()
model.load_state_dict(torch.load('jt1_pos/lstm_model_epoch149.pth', map_location=torch.device('cpu')))
# model = torch.jit.load('/home/cerebro/snyder_project/SRALabProject/misc/model_scripting/scripts/lstm_trial5.pt')
model.eval()

# Testing
with torch.no_grad():
    y_pred = model(X_test)
    loss_fn = nn.MSELoss()
    test_rmse = torch.sqrt(loss_fn(y_pred, y_test))
    print(f"Testing RMSE: {test_rmse:.4f}")

with torch.no_grad():
    # Get all predictions (taking last timestep from each sequence)
    y_pred2 = model(X_test)[:, -1, :].squeeze()  # Shape: [n_samples]
    y_true = y_test[:, -1, :].squeeze()        # Shape: [n_samples]
    
    # # Pointwise absolute errors
    # errors = torch.abs(y_pred2 - y_true).numpy()  # Convert to numpy for plotting
    # Pointwise errors (signed)
    errors = (y_pred2 - y_true).numpy()  # Convert to numpy for plotting
    print(f'Shape of Errors: {errors.shape}')

# Plotting Prep
with torch.no_grad():
    # Create plotting arrays
    plot_length = len(test)  # Only plot testing portion
    
    # Fill true values (offset by lookback)
    therapist_true = test[:, predicted_target]
    therapist_true = therapist_true.reshape(-1, 1)
    
    # Get predictions (last timestep of each sequence)
    test_pred = model(X_test)[:, -1, :].numpy()

    # Fill predictions (aligned with true values)
    therapist_pred = np.full_like(therapist_true, np.nan)
    therapist_pred[lookback:lookback + len(test_pred)] = test_pred

# Find peaks in the data and divide into periodic segments
data_peaks0, _ = find_peaks(therapist_true[:,0], height=0.4, distance=1000)
periodic_data0 = [therapist_true[data_peaks0[i]:data_peaks0[i+1],0] for i in range(len(data_peaks0)-1)]
pred_peaks0, _ = find_peaks(therapist_pred[:,0], height=0.4, distance=1000)
periodic_pred0 = [therapist_pred[pred_peaks0[i]:pred_peaks0[i+1],0] for i in range(len(pred_peaks0)-1)]

# Normalize periodic data
normalized_length = 101
normalized_periodic_data0 = []
normalized_periodic_pred0 = []
for period in periodic_data0:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_data0.append(norm_period)
for period in periodic_pred0:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_pred0.append(norm_period)

# Get mean and std dev of normalized data and pred
stacked_data0 = np.vstack(normalized_periodic_data0)
stacked_pred0 = np.vstack(normalized_periodic_pred0)
mean_data0 = np.mean(stacked_data0, axis=0)
mean_pred0 = np.mean(stacked_pred0, axis=0)
std_data0 = np.std(stacked_data0, axis=0)
std_pred0 = np.std(stacked_pred0, axis=0)
print(f'Stacked Data0 Size: {stacked_data0.shape}')
print(f'Stacked Pred0 Size: {stacked_pred0.shape}')

# # Get errors for periodic data
# # Must stay commented out until periodic data is set correctly
mean_err0 = np.mean(abs(stacked_data0 - stacked_pred0), axis=0)
std_err0 = np.std(abs(stacked_data0 - stacked_pred0), axis=0)

# Create two synchronized animations: patient data and predictions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
plt.subplots_adjust(hspace=0.4)

# Configuration parameters - ADJUST THESE AS NEEDED
TIME_WINDOW_SECONDS = 15  # Total time to show in animation (seconds)
SAMPLING_RATE_HZ = 333    # Data sampling rate (Hz) - adjust based on your data
ANIMATION_INTERVAL = 3   # Milliseconds between frames

# Calculate data points based on time
total_data_points = int(TIME_WINDOW_SECONDS * SAMPLING_RATE_HZ)
print(f"Showing {TIME_WINDOW_SECONDS} seconds of data ({total_data_points} points)")

# Set up the patient data plot (top)
ax1.set_xlim(0, total_data_points)  # Change to time steps
ax1.set_ylim(np.min(test[:, pIndex]) - 0.1, np.max(test[:, pIndex]) + 0.1)
ax1.set_ylabel('Joint Position (Radians)', fontsize=20)
ax1.set_title('Patient Joint Data', fontsize=20)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_xlabel('Time Steps (~3ms)', fontsize=20)
ax1.grid(True)

# Set up the prediction plot (bottom)
ax2.set_xlim(0, total_data_points)  # Change to time steps
ax2.set_ylim(np.min(therapist_true) - 0.1, np.max(therapist_true) + 0.1)
ax2.set_ylabel('Joint Position (Radians)', fontsize=20)
ax2.set_title('Prediction', fontsize=20)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_xlabel('Time Steps (~3ms)', fontsize=20)
ax2.grid(True)

# Initialize empty lines for patient data
line_patient, = ax1.plot([], [], 'g-', linewidth=2, label=f'Patient Joint {pIndex+1}')

# Initialize empty lines for predictions
line_pred, = ax2.plot([], [], 'r-', linewidth=2, label='Predicted Therapist Data')

# Prepare the raw data for streaming (no normalization)
# Use the first total_data_points of each dataset
patient_data_raw = test[:total_data_points, pIndex]

# For predictions, we need to handle NaN values at the beginning
pred_data_raw = therapist_pred[:total_data_points, 0]

# Create time step array (not seconds)
time_steps = np.arange(total_data_points)  # This will be 0, 1, 2, 3, ... total_data_points-1

# Animation update function for both plots
def update_synchronized(frame):
    """Update function for synchronized animation of patient data and predictions.
    
    Args:
        frame (int): Current frame index.
    Returns:
        tuple: Updated line objects for patient data and predictions.
    """
    # Calculate how many points to show (growing from left to right)
    points_to_show = min(frame + 1, total_data_points)
    
    if points_to_show > 0:
        current_time_steps = time_steps[:points_to_show]
        
        # Update patient data plot
        patient_data = patient_data_raw[:points_to_show]
        line_patient.set_data(current_time_steps, patient_data)
        
        # Update prediction plot
        pred_data = pred_data_raw[:points_to_show]
        line_pred.set_data(current_time_steps, pred_data)
    
    return line_patient, line_pred

# Create synchronized animation
ani_sync = animation.FuncAnimation(
    fig, 
    update_synchronized, 
    frames=total_data_points,
    interval=ANIMATION_INTERVAL,
    blit=True,
    repeat=True
)
plt.show()

# ALL PLOTTING HAPPENS BELOW HERE. READ INSTRUCTIONS FOR CREATING BEST PLOTS
# Plot histogram of errors
# fig = plt.figure(figsize=(12, 6), )
# plt.hist(errors, bins=50, alpha=0.7, color='blue')
# # plt.xlabel('Error (Radians)')
# plt.xlabel('Error (Radians/second)')
# plt.ylabel('# of Occurrences')
# # plt.title(f'Therapist Joint 2 Position Errors, RMSE: {test_rmse:.4f}, Max (abs): {abs(errors).max():.4f}, Std Dev (abs): {abs(errors).std():.4f}, Mean (abs): {abs(errors).mean():.4f}')
# plt.title(f'Therapist Joint 1 Velocity Errors, RMSE: {test_rmse:.4f}, Max (abs): {abs(errors).max():.4f}, Std Dev (abs): {abs(errors).std():.4f}, Mean (abs): {abs(errors).mean():.4f}')
# plt.grid(True)
# # plt.savefig('/home/gis/Documents/SRALabProject/lstm_FullData/plots/trial9/jt3_vel_error_histogram.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
# plt.show()

# use to determining amplitude for periodic plots
# fig = plt.figure(figsize=(12, 6))
# joint_pairs = [[0, 2], [1, 3], [2, 0], [3, 1]]  # Pairs of joints to plot together
# plt.plot(therapist_true[:, 1], c='b', label=f'True Therapist Data')
# plt.plot(therapist_pred[:, 1], c='r', linestyle='--', label=f'Predicted Therapist Data')
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions (Radians)')
# plt.legend()
# plt.show()

# Plot only testing data with prediction overlay
# Change xlim for desired time steps
# Limit to 7500 time steps (~30 seconds) EXCEPT when determining amplitude for periodic plots
# fig = plt.figure(figsize=(12, 6))
# # plt.title(f"Therapist Joint 2 Position Predictions with Patient Joint 4")
# plt.title(f"Therapist Joint 1 Velocity Predictions with Patient Joint 3")
# plt.plot(test[:, pIndex], c='g', label=f'Patient Data') # Set to corresponding patient data index
# plt.plot(therapist_true, c='b', label=f'True Therapist Data')
# plt.plot(therapist_pred, c='r', linestyle='--', label=f'Predicted Therapist Data')
# # plt.scatter(np.arange(len(therapist_pred)), therapist_pred[:, i], c='r', marker='x', label=f'Predicted Therapist Data', alpha=0.5)
# plt.xlim(0, 7500)
# plt.xlabel('Time Steps (~3ms)')
# # plt.ylabel('Joint Positions (Radians)')
# plt.ylabel('Joint Velocities (Radians/second)')
# plt.legend()
# # plt.savefig('/home/gis/Documents/SRALabProject/lstm_FullData/plots/trial9/jt3_vel_predictions.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
# plt.show()

# # Plot error over time
# plt.figure(figsize=(12, 6))
# plt.plot(errors, c='r', label='Prediction Error')
# plt.xlim(0, 7500)
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions Errors (Radians)')
# plt.legend()
# plt.title("Error over Time")
# plt.show()

# Plot normalized periodic data
fig = plt.figure(figsize=(12, 6))

# Plots each period as a separate line. Use to look for outliers in amplitude
# Missteps can cause outliers where multiple periods will be combined into one. Need to fix height in 'find peaks' above
# for period in normalized_periodic_data1:
#     plt.plot(period, c='b', alpha=0.5)
# for period in normalized_periodic_pred1:
#     plt.plot(period, c='r', alpha=0.5)

# Plot mean and std dev of periodic data and predictions
# Use after height is set correctly in 'find peaks' above
plt.plot(mean_data0, c='b', label='Mean Therapist Data')
plt.fill_between(range(normalized_length), 
                 mean_data0 - std_data0, 
                 mean_data0 + std_data0, 
                 color='b', alpha=0.2)
plt.plot(mean_pred0, c='r', label='Mean Predicted Therapist Data')
plt.fill_between(range(normalized_length), 
                 mean_pred0 - std_pred0, 
                 mean_pred0 + std_pred0, 
                 color='r', alpha=0.2)
plt.title("Left Hip")
plt.ylabel('Joint Positions (Radians)')
plt.xlabel('Gait Phase %')
plt.legend()

# Plot period plot of error using mean and std dev
# plt.plot(mean_err0, c='r', label='Mean Error')
# plt.fill_between(range(normalized_length), 
#                  mean_err0 - std_err0, 
#                  mean_err0 + std_err0, 
#                  color='r', alpha=0.2)
# plt.title("Left Hip")
# plt.ylabel('Joint Error (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()

# plt.show()
