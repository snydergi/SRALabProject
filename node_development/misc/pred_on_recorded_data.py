import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

# Load data
datapath = '/home/cerebro/snyder_project/SRALabProject/node_development/misc/pred_data/prediction_1752856570.5665696.csv'
data = pd.read_csv(datapath, header=None)

# Get input data (last 11 columns)
data_input = data.iloc[:, -11:].values.astype(np.float32)

# Model definition
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 4)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)

# Load model
model = JointModel()
model.load_state_dict(torch.load('/home/cerebro/snyder_project/SRALabProject/lstm_BigData/trial5/lstm_model_epoch198.pth'))
model.eval()

# Generate predictions
window_size = 50  # Should match your training sequence length
predictions = []

with torch.no_grad():
    for i in range(len(data_input) - window_size + 1):
        window = data_input[i:i+window_size]
        input_tensor = torch.tensor(window).unsqueeze(0)  # Add batch dimension
        pred = model(input_tensor)[:, -1, :].numpy()
        predictions.append(pred.squeeze())

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