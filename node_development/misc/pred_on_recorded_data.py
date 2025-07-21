import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

# Load data
datapath = '/home/gis/SRALab_Data/ExperimentDay1/prediction_1752606480.2458458.csv'
data = pd.read_csv(datapath, header=None)

# Get input data (last 11 columns)
data_input = data.iloc[:, -11:].values.astype(np.float32)
print("Data Input Shape:", data_input.shape)

# Model definition
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 4)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)

# Create dataset
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series data
        lookback: Size of window for prediction
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
model.load_state_dict(torch.load('/home/gis/Documents/SRALabProject/lstm_BigData/trial5/lstm_model_epoch198.pth'))
model.eval()

# Generate predictions
lookback = 50
X_test, y_test = create_dataset(data_input, lookback)
predictions = []

with torch.no_grad():
    # Get all predictions (taking last timestep from each sequence)
    y_pred2 = model(X_test)[:, -1, :].squeeze()  # Shape: [n_samples]
    y_true = y_test[:, -1, :].squeeze()        # Shape: [n_samples]

predictions = np.array(predictions)

# Create scatter plots
plt.figure(figsize=(12, 8))

# Create time axis for x-values
time_steps = np.arange(len(predictions))

# Define colors and markers for each joint
joint_styles = [
    {'color': 'blue', 'marker': 'o', 'label': 'Joint 1'},
    {'color': 'red', 'marker': 's', 'label': 'Joint 2'},
    {'color': 'green', 'marker': '^', 'label': 'Joint 3'},
    {'color': 'purple', 'marker': 'D', 'label': 'Joint 4'}
]

# Plot each joint's predictions
for i in range(4):
    plt.scatter(
        time_steps,
        predictions[:, i],
        c=joint_styles[i]['color'],
        marker=joint_styles[i]['marker'],
        label=joint_styles[i]['label'],
        alpha=0.6,
        edgecolors='w',
        linewidths=0.5
    )

plt.title('Predicted Joint Angles (Scatter Plot)', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Angle (rad)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()