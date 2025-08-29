"""Another testing program for the stacked LSTM model, this time using scripted model loading."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import time

# Set plotting parameters
plt.rcParams['svg.fonttype'] = 'none'

# Load the data
# DATA ORDER:
# p_jt_pos1, p_jt_pos2, p_jt_pos3, p_jt_pos4, p_jt_vel1, p_jt_vel2, p_jt_vel3, p_jt_vel4, p_SIF, p_bp_pos, p_bp_vel,
# t_jt_pos1, t_jt_pos2, t_jt_pos3, t_jt_pos4, t_jt_vel1, t_jt_vel2, t_jt_vel3, t_jt_vel4
test_1 = pd.read_csv('../../lstm_FullData/data/test_set_1.csv', skiprows=0).values
test_2 = pd.read_csv('../../lstm_FullData/data/test_set_2.csv', skiprows=0).values
test_3 = pd.read_csv('../../lstm_FullData/data/test_set_3.csv', skiprows=0).values

def create_dataset(dataset, lookback, step=1):
    """Transform time series data into a prediction dataset.
    
    Args:
        dataset (np.ndarray): An array of time series data. Shape [timesteps, features]
        lookback (int): Size of window for prediction
        step (int, optional): Step between consecutive windows. Defaults to 1.

    Returns:
        tuple:
            - X (torch.Tensor): Feature tensor, shape [samples, lookback, 8]
            - y (torch.Tensor): Target tensor, shape [samples, lookback, 8]

    Notes:
        - Change second feature index to select feature joint/joints
        - Change second target index to select target joint/joints
    """
    X, y = [], []
    for i in range(0, len(dataset)-lookback, step):
        feature = dataset[i:i+lookback, :8]  # Feature is patient data, if input size 8
        target = dataset[i+1:i+lookback+1, -8:]  # Target is therapist data
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lookback, step = 50, 5
patient_number = 1 # Set based on which patient data is being tested
test = test_1 if patient_number == 1 else test_2 if patient_number == 2 else test_3
predicted_target = -2 # Index based on data order above
pIndex = 4 # Index of patient joint to plot against therapist joint (0-3 for patient joints 1-4)
X_test, y_test = create_dataset(test, lookback=lookback)

# Load model
model = torch.jit.load('../model_scripting/scripts/stacked_model.pt')
model.eval()
loss_fn = nn.MSELoss()

# Create DataLoader
loader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=False, batch_size=1)

# Testing
with torch.no_grad():
    test_rmse = []
    pred_times = []
    # Iterate through DataLoader
    for X_batch, y_batch in loader:
        pred_start = time.time()
        y_pred = model(X_batch)
        pred_end = time.time()
        pred_times.append(pred_end - pred_start)
        # print(f'Prediction time for batch: {pred_times[-1]:.6f} seconds')

print(f'Average prediction time per batch: {np.mean(pred_times):.6f} seconds')
