import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Load the first range (164041 to 367374)
patient_part1 = pd.read_csv('X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 
                          skiprows=range(1, 164041), 
                          nrows=367374-164041)
therapist_part1 = pd.read_csv('X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 
                            skiprows=range(1, 164041), 
                            nrows=367374-164041)

# Load the second range (454139 to 654139)
patient_part2 = pd.read_csv('X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 
                          skiprows=range(1, 454139), 
                          nrows=654139-454139)
therapist_part2 = pd.read_csv('X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 
                            skiprows=range(1, 454139), 
                            nrows=654139-454139)

# Concatenate the two parts
patient = pd.concat([patient_part1, patient_part2])
therapist = pd.concat([therapist_part1, therapist_part2])

patient_data = patient[['JointPositions_1']].values.astype('float32')
therapist_data = therapist[['JointPositions_1']].values.astype('float32')

timeseries = np.column_stack((patient_data, therapist_data))
 
# plt.plot(timeseries)
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions')
# plt.title('Left Hip Joint Positions Over Time')
# plt.legend(['Patient', 'Therapist'])
# plt.xlim(0, 7500)
# plt.show()

# train-test split for time series
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, 0]  # Feature is patient data
        target = dataset[i+1:i+lookback+1, 1]  # Target is therapist data
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)

lookback = 50  # ~4ms timestep, so ~200ms lookback window
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        # extract only the last time step
        # x = x[:, -1, :]
        x = self.linear(x)
        return x
    

model = JointModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=256)

train_loss_list = []
test_loss_list = []
n_epochs = 500
for epoch in range(n_epochs):
    model.train()
    epoch_losses = []
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    train_loss_list.append(np.mean(epoch_losses))
    # Validation
    # if epoch % 100 != 0:
    #     continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        test_loss_list.append(loss_fn(y_pred, y_test))
        print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, Test RMSE {test_rmse:.4f}")

# Plotting the losses
plt.figure(figsize=(12, 6))
plt.plot(train_loss_list, label='Training Loss', alpha=0.8)
plt.plot(test_loss_list, label='Test Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

# Add a vertical line at the best test loss point
best_epoch = np.argmin(test_loss_list)
plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
plt.legend()

# Save the plot
plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')

torch.save(model.state_dict(), 'lstm_model.pth')

# Plotting
with torch.no_grad():
    # Create arrays for plotting
    therapist_true = np.ones_like(timeseries[:, 1]) * np.nan
    therapist_pred = np.ones_like(timeseries[:, 1]) * np.nan
    
    # Fill in the true therapist data
    therapist_true[lookback:] = timeseries[lookback:, 1]
    
    # Get predictions for train and test
    train_pred = model(X_train)[:, -1, :].numpy().flatten()
    test_pred = model(X_test)[:, -1, :].numpy().flatten()
    
    # Fill in the predicted therapist data
    therapist_pred[lookback:train_size] = train_pred
    therapist_pred[train_size+lookback:] = test_pred

# plot
plt.figure(figsize=(12, 6))
plt.plot(therapist_true, c='b', label='True Therapist Data')
plt.plot(therapist_pred, c='r', linestyle='--', label='Predicted Therapist Data')
plt.plot(timeseries[:, 0], c='g', alpha=0.3, label='Patient Data (input)')
plt.xlim(0, 7500)
plt.xlabel('Time Steps (~4ms)')
plt.ylabel('Joint Positions')
plt.legend()
plt.title("Therapist Hip Data Prediction from Patient Data")
plt.show()
plt.savefig('lstm_therapist_prediction.png', dpi=300, bbox_inches='tight')