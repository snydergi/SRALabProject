import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

# Load data
patient = pd.read_csv('X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 
                     skiprows=range(1, 696306), 
                     nrows=894640-696306)
therapist = pd.read_csv('X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 
                       skiprows=range(1, 696306), 
                       nrows=894640-696306)

# Prepare data
patient_data = patient[['JointPositions_2']].values.astype('float32')
therapist_data = therapist[['JointPositions_4']].values.astype('float32')
timeseries = np.column_stack((patient_data, therapist_data))

# Validation split
valid_size = int(len(timeseries) * 0.5)
valid = timeseries[(len(timeseries) - valid_size):len(timeseries)]

# Create dataset function (optimized)
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset"""
    X = np.array([dataset[i:i+lookback, 0] for i in range(len(dataset)-lookback)])
    y = np.array([dataset[i+1:i+lookback+1, 1] for i in range(len(dataset)-lookback)])
    return (torch.tensor(X).unsqueeze(-1), 
            torch.tensor(y).unsqueeze(-1))

lookback = 50
X_valid, y_valid = create_dataset(valid, lookback=lookback)

# Model definition
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)

# Load model
model = JointModel()
model.load_state_dict(torch.load('trial5/j24/lstm_model_epoch139.pth'))
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
plt.title(f'Distribution of Prediction Errors, RMSE: {valid_rmse:.4f}, Max: {errors.max():.4f}, Std Dev: {errors.std():.4f}, Mean: {errors.mean():.4f}')
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
plt.figure(figsize=(12, 6))
plt.plot(therapist_true, c='b', label='True Therapist Data')
plt.plot(therapist_pred, c='r', linestyle='--', label='Predicted Therapist Data')
# plt.plot(valid[:, 0], c='g', alpha=0.75, label='Patient Data (input)')
plt.xlim(0, 7500)
plt.xlabel('Time Steps (~4ms)')
plt.ylabel('Joint Positions (Radians)')
plt.legend()
plt.title("Validation: Therapist Knee Prediction from Patient Data")
plt.show()
# plt.savefig('lstm_therapist_prediction.png', dpi=300, bbox_inches='tight')

# Plot error over time
plt.figure(figsize=(12, 6))
plt.plot(errors, c='r', label='Prediction Error')
plt.xlim(0, 7500)
plt.xlabel('Time Steps (~4ms)')
plt.ylabel('Joint Positions Errors (Radians)')
plt.legend()
plt.title("Error over Time")
plt.show()