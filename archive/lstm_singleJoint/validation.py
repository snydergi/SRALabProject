import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Load data
patient = pd.read_csv('X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 
                     skiprows=range(1, 696306), 
                     nrows=894640-696306)
therapist = pd.read_csv('X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 
                       skiprows=range(1, 696306), 
                       nrows=894640-696306)

# Prepare data
patient_data = patient[['JointPositions_1']].values.astype('float32')
therapist_data = therapist[['JointPositions_3']].values.astype('float32')
timeseries = np.column_stack((patient_data, therapist_data))

# Validation split
valid_size = int(len(timeseries) * 0.5)
valid = timeseries[(len(timeseries) - valid_size):len(timeseries)]

# Create dataset function
def create_dataset(dataset, lookback):
    """Transform time series data into a prediction dataset.
    
    Args:
        dataset (np.ndarray): An array of time series data. Shape [timesteps, features]
        lookback (int): Size of window for prediction

    Returns:
        tuple:
            - X (torch.Tensor): Feature tensor, shape [samples, lookback]
            - y (torch.Tensor): Target tensor, shape [samples, lookback]

    Notes:
        - Change second feature index to select feature joint/joints
        - Change second target index to select target joint/joints
    """
    X = np.array([dataset[i:i+lookback, 0] for i in range(len(dataset)-lookback)])
    y = np.array([dataset[i+1:i+lookback+1, 1] for i in range(len(dataset)-lookback)])
    return (torch.tensor(X).unsqueeze(-1), 
            torch.tensor(y).unsqueeze(-1))

lookback = 50
X_valid, y_valid = create_dataset(valid, lookback=lookback)

# Model definition
class JointModel(nn.Module):
    """A neural network model combining LSTM and linear layers for sequence prediction.
    
    This model processes sequential input data using an LSTM layer followed by a linear
    transformation layer to produce predictions for each timestep.
    
    Architecture:
        - LSTM layer: 1 layer, 50 hidden units, processes sequences with 1 input features
        - Linear layer: Maps from 50-dimensional LSTM output to 1-dimensional prediction
    """
    def __init__(self):
        """Initialize JointModel."""
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, 4]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, 1]
        """
        x, _ = self.lstm(x)
        return self.linear(x)

# Load model
model = JointModel()
model.load_state_dict(torch.load('trial6/lstm_model_epoch70.pth'))
model.eval()

# Validation
with torch.no_grad():
    y_pred = model(X_valid)
    loss_fn = nn.MSELoss()
    valid_rmse = torch.sqrt(loss_fn(y_pred, y_valid))
    print(f"Validation RMSE: {valid_rmse:.4f}")

with torch.no_grad():
    # Get all predictions (taking last timestep from each sequence)
    y_pred2 = model(X_valid)[:, -1, :].squeeze()  # Shape: [n_samples]
    y_true = y_valid[:, -1, :].squeeze()        # Shape: [n_samples]
    
    # # Pointwise absolute errors
    # errors = torch.abs(y_pred2 - y_true).numpy()  # Convert to numpy for plotting
    # Pointwise errors (signed)
    errors = (y_pred2 - y_true).numpy()  # Convert to numpy for plotting

# Plot histogram of errors
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50, alpha=0.7, color='blue')
plt.xlabel('Error (Radians)')
plt.ylabel('# of Occurrences')
plt.title(f'1-to-1 (R Hip) Distribution of Prediction Errors, RMSE: {valid_rmse:.4f}, Max (abs): {abs(errors).max():.4f}, Std Dev (abs): {abs(errors).std():.4f}, Mean (abs): {abs(errors).mean():.4f}')
plt.grid(True)
plt.show()

# Plotting
with torch.no_grad():
    # Create plotting arrays
    plot_length = len(valid)  # Only plot validation portion
    therapist_true = np.full(plot_length, np.nan)
    therapist_pred = np.full(plot_length, np.nan)
    
    # Fill true values (offset by lookback)
    therapist_true[lookback:] = valid[lookback:, 1]
    
    # Get predictions (last timestep of each sequence)
    valid_pred = model(X_valid)[:, -1, :].numpy().flatten()
    
    # Fill predictions (aligned with true values)
    therapist_pred[lookback:lookback+len(valid_pred)] = valid_pred

# Plot only validation portion
# plt.figure(figsize=(12, 6))
# plt.plot(therapist_true, c='b', label='True Therapist Data')
# # plt.plot(therapist_pred, c='r', linestyle='--', label='Predicted Therapist Data')
# # plt.plot(valid[:, 0], c='g', alpha=0.75, label='Patient Data (input)')
# plt.xlim(0, 7500)
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions (Radians)')
# plt.legend()
# plt.title("Validation: Therapist Hip Prediction from Patient Data")
# plt.show()
# plt.savefig('lstm_therapist_prediction.png', dpi=300, bbox_inches='tight')

# # Plot error over time
# plt.figure(figsize=(12, 6))
# plt.plot(errors, c='r', label='Prediction Error')
# plt.xlim(0, 7500)
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions Errors (Radians)')
# plt.legend()
# plt.title("Error over Time")
# plt.show()


# Overlay of periodic data
# Find peaks in the therapist data and divide into periodic segments
data_peaks, _ = find_peaks(therapist_true, height=0.9, distance=1000)  # ONLY NEGATIVE FOR KNEES (2,4), Heights: J13=1.0, J24=1.4, J31=0.4, J42=1.0
periodic_data = [therapist_true[data_peaks[i]:data_peaks[i+1]] for i in range(len(data_peaks)-1)]
pred_peaks, _ = find_peaks(therapist_pred, height=0.9, distance=1000)  # ONLY NEGATIVE FOR KNEES (2,4), Heights: J13=1.0, J24=1.4, J31=0.4, J42=1.0
periodic_pred = [therapist_pred[pred_peaks[i]:pred_peaks[i+1]] for i in range(len(pred_peaks)-1)]

# Normalize periodic data
normalized_length = 101
normalized_periodic_data = []
normalized_periodic_pred = []
for period in periodic_data:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_data.append(norm_period)
for period in periodic_pred:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_pred.append(norm_period)

# Get mean and std dev of normalized data and pred
stacked_data = np.vstack(normalized_periodic_data)
stacked_pred = np.vstack(normalized_periodic_pred)
mean_data = np.mean(stacked_data, axis=0)
mean_pred = np.mean(stacked_pred, axis=0)
std_data = np.std(stacked_data, axis=0)
std_pred = np.std(stacked_pred, axis=0)

# Plot normalized periodic data
plt.figure(figsize=(12, 6))
# for period in normalized_periodic_data:
#     plt.plot(period, c='b', alpha=0.5)
# for period in normalized_periodic_pred:
#     plt.plot(period, c='r', alpha=0.5)
plt.plot(mean_data, c='b', label='Mean Therapist Data')
plt.fill_between(range(normalized_length), 
                 mean_data - std_data, 
                 mean_data + std_data, 
                 color='b', alpha=0.2)
plt.plot(mean_pred, c='r', label='Mean Predicted Therapist Data')
plt.fill_between(range(normalized_length), 
                 mean_pred - std_pred, 
                 mean_pred + std_pred, 
                 color='r', alpha=0.2)
plt.legend()
plt.title("Average Periodic Data and Prediction, Therapist Right Hip")
plt.xlabel('Normalized Time Steps')
plt.ylabel('Joint Positions (Radians)')
plt.show()
