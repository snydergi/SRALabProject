import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series data
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, :8]  # Feature is patient data
        target = dataset[i+1:i+lookback+1, -8]  # Target is therapist data
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)

lookback = 50  # ~4ms timestep, so ~200ms lookback window
X_train, y_train = create_dataset(train, lookback=lookback)
X_valid, y_valid = create_dataset(valid, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 4)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    

model = JointModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# optimizer = optim.Adadelta(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=256)

train_loss_list = []
valid_loss_list = []
n_epochs = 150
for epoch in range(n_epochs):
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
    with torch.no_grad():
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_valid, y_valid = X_valid.to(device), y_valid.to(device)
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_valid)
        valid_rmse = np.sqrt(loss_fn(y_pred, y_valid))
        valid_loss_list.append(loss_fn(y_pred, y_valid))
        print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, Validation RMSE {valid_rmse:.4f}")
    torch.save(model.state_dict(), f'lstm_model_epoch{epoch}.pth')

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
plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')

torch.save(model.state_dict(), 'lstm_model.pth')
