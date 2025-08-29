"""Check predictions on recorded data and plot each joint in separate subplots."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import csv

# Load data
datapath = '/home/gis/SRALab_Data/ExperimentDay3/prediction_1753215345.0074022.csv'
data = pd.read_csv(datapath, header=None)
data = data[data[0].str.contains('trial4')]
# Get input data (last 11 columns)
data_input = data.iloc[:, -11:].values.astype(np.float32)
print("Data Input Shape:", data_input.shape)

# Model definition
class JointModel(nn.Module):
    """A neural network model combining LSTM and linear layers for sequence prediction.
    
    This model processes sequential input data using an LSTM layer followed by a linear
    transformation layer to produce predictions for each timestep.
    
    Architecture:
        - LSTM layer: 1 layer, 50 hidden units, processes sequences with 11 input features
        - Linear layer: Maps from 50-dimensional LSTM output to 4-dimensional prediction
    """
    def __init__(self):
        """Initialize JointModel."""
        super().__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 4)
    
    def forward(self, x):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, 11]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, 4]
        """
        x, _ = self.lstm(x)
        return self.linear(x)

# Create dataset
def create_dataset(dataset, lookback):
    """Transform time series data into a prediction dataset.
    
    Args:
        dataset (np.ndarray): An array of time series data. Shape [timesteps, features]
        lookback (int): Size of window for prediction

    Returns:
        tuple:
            - X (torch.Tensor): Feature tensor, shape [samples, lookback, 11]
            - y (torch.Tensor): Target tensor, shape [samples, lookback, 4]

    Notes:
        - Change second feature index to select feature joint/joints
        - Change second target index to select target joint/joints
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, :11]  # Feature is patient data, change based on input size
        target = dataset[i+1:i+lookback+1, -4:]  # Target is therapist data
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)

# Load model
model = JointModel()
model.load_state_dict(torch.load('/home/gis/Documents/SRALabProject/lstm_FullData/trial5/lstm_model_epoch198.pth'))
model.eval()

# Generate predictions
lookback = 50
X_test, y_test = create_dataset(data_input, lookback)
predictions = []

with torch.no_grad():
    # Get all predictions (taking last timestep from each sequence)
    y_pred2 = model(X_test)[:, -1, :].squeeze()  # Shape: [n_samples]
    y_true = y_test[:, -1, :].squeeze()        # Shape: [n_samples]

predictions = np.array(y_pred2)

# Create time axis for x-values
time_steps = np.arange(len(predictions))

# Define colors and markers for each joint
joint_styles = [
    {'color': 'blue', 'marker': 'o', 'label': 'Joint 1'},
    {'color': 'red', 'marker': 's', 'label': 'Joint 2'},
    {'color': 'green', 'marker': '^', 'label': 'Joint 3'},
    {'color': 'purple', 'marker': 'D', 'label': 'Joint 4'}
]

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 columns
fig.suptitle('Predicted Joint Angles (Scatter Plots)', fontsize=14)

# Flatten the axes array for easy iteration
axes = axes.ravel()

# Plot each joint's predictions in its own subplot
for i in range(4):
    axes[i].plot(
        time_steps,
        predictions[:, i],
        color=joint_styles[i]['color'],
        label=joint_styles[i]['label'],
        alpha=0.6,
        linewidth=0.5
    )
    # axes[i].scatter(
    #     time_steps,
    #     predictions[:, i],
    #     c=joint_styles[i]['color'],
    #     marker=joint_styles[i]['marker'],
    #     label=joint_styles[i]['label'],
    #     alpha=0.6,
    #     edgecolors='w',
    #     linewidths=0.5
    # )
    
    axes[i].set_title(joint_styles[i]['label'])
    axes[i].set_xlabel('Time Step', fontsize=10)
    axes[i].set_ylabel('Angle (rad)', fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.show()
