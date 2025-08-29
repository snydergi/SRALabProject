#### This structure was used for each model in trial 9, only changing input specifications.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the data
patient_data = pd.read_csv('../../data/patient_training_full.csv', skiprows=0).values
therapist_data = pd.read_csv('../../data/therapist_training_full.csv', skiprows=0).values
patient_valid = pd.read_csv('../../data/patient_validation_full.csv', skiprows=0).values
therapist_valid = pd.read_csv('../../data/therapist_validation_full.csv', skiprows=0).values

# DATA ORDER:
# p_jt_pos1, p_jt_pos2, p_jt_pos3, p_jt_pos4, p_jt_vel1, p_jt_vel2, p_jt_vel3, p_jt_vel4, p_SIF, p_bp_pos, p_bp_vel,
# t_jt_pos1, t_jt_pos2, t_jt_pos3, t_jt_pos4, t_jt_vel1, t_jt_vel2, t_jt_vel3, t_jt_vel4
timeseries = np.column_stack((patient_data, therapist_data))
timeseries_valid = np.column_stack((patient_valid, therapist_valid))

# Should be (x, 19)
print(f"Training data shape: {timeseries.shape}")
print(f"Validation data shape: {timeseries_valid.shape}")

# train-valid split for time series
train, valid = timeseries, timeseries_valid

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
    for i in range(0, len(dataset)-lookback, step):
        feature = dataset[i:i+lookback, :8]  # Feature is patient data, shape [lookback, 8]
        target = dataset[i+1:i+lookback+1, -8]  # Target is therapist data, shape [lookback]
        target = target.reshape(-1, 1) # Reshape target to be [lookback, 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lookback, step = 50, 1  # ~4ms timestep, so ~200ms lookback window
X_train, y_train = create_dataset(train, lookback=lookback)
X_valid, y_valid = create_dataset(valid, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

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
        x = self.linear(x)
        return x
    

model = JointModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# optimizer = optim.Adadelta(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=256)
loader_val = data.DataLoader(data.TensorDataset(X_valid, y_valid), shuffle=True, batch_size=256)

train_loss_list = []
valid_loss_list = []
n_epochs = 150
for epoch in range(n_epochs):
    start_time = time.time()
    model.train()
    epoch_losses = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_loss_list.append(np.mean(epoch_losses))
    model.eval()
    epoch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in loader_val:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            epoch_val_losses.append(loss.item())
        valid_loss_list.append(np.mean(epoch_val_losses))
        print(f"Epoch {epoch}: Train MSE {np.mean(epoch_losses):.4f}, Validation MSE {np.mean(epoch_val_losses):.4f}")
    torch.save(model.state_dict(), f'lstm_model_epoch{epoch}.pth')
    print(f"Epoch {epoch} training time: {time.time() - start_time}")

# Plotting the losses
plt.figure(figsize=(12, 6))
plt.plot(train_loss_list, label='Training Loss', alpha=0.8)
plt.plot(valid_loss_list, label='Validation Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

# Add a vertical line at the best valid loss point
best_epoch = np.argmin(valid_loss_list)
plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
plt.legend()

# Save the plot
plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight', format='svg')

torch.save(model.state_dict(), 'lstm_model.pth')
