import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# First Subject-Therapist Pair
patient1_datapath = '../data/Patient1_X2_SRA_A_07-05-2024_10-39-10.csv'
therapist1_datapath = '../data/Therapist1_X2_SRA_B_07-05-2024_10-41-46.csv'
patient1_part1 = pd.read_csv(patient1_datapath, 
                          skiprows=range(1, 229607), 
                          nrows=433021-229607)
therapist1_part1 = pd.read_csv(therapist1_datapath, 
                            skiprows=range(1, 165636), 
                            nrows=369050-165636)
patient1_part2 = pd.read_csv(patient1_datapath, 
                          skiprows=range(1, 516255), 
                          nrows=718477-516255)
therapist1_part2 = pd.read_csv(therapist1_datapath, 
                            skiprows=range(1, 452284), 
                            nrows=654506-452284)
patient1_part3 = pd.read_csv(patient1_datapath, 
                          skiprows=range(1, 761002), 
                          nrows=960356-761002)
therapist1_part3 = pd.read_csv(therapist1_datapath, 
                            skiprows=range(1, 697032), 
                            nrows=896386-697032)

# Second Subject-Therapist Pair
patient2_datapath = '../data/Patient2_X2_SRA_A_08-05-2024_14-33-44.csv'
therapist2_datapath = '../data/Therapist2_X2_SRA_B_08-05-2024_14-33-51.csv'
patient2_part1 = pd.read_csv(patient2_datapath,
                            skiprows=range(1, 123920),
                            nrows=272301-123920)
therapist2_part1 = pd.read_csv(therapist2_datapath,
                            skiprows=range(1, 123920),
                            nrows=272301-123920)

# Third Subject-Therapist Pair
patient3_datapath = '../data/Patient3_X2_SRA_A_29-05-2024_13-36-40.csv'
therapist3_datapath = '../data/Therapist3_X2_SRA_B_29-05-2024_13-41-19.csv'
patient3_part1 = pd.read_csv(patient3_datapath, 
                          skiprows=range(1, 7694), 
                          nrows=198762-7694)
therapist3_part1 = pd.read_csv(therapist3_datapath, 
                            skiprows=range(1, 7694), 
                            nrows=198762-7694)
patient3_part2 = pd.read_csv(patient3_datapath, 
                          skiprows=range(1, 232681), 
                          nrows=388280-232681)
therapist3_part2 = pd.read_csv(therapist3_datapath, 
                            skiprows=range(1, 232681), 
                            nrows=388280-232681)
patient3_part3 = pd.read_csv(patient3_datapath, 
                          skiprows=range(1, 417061), 
                          nrows=606461-417061)
therapist3_part3 = pd.read_csv(therapist3_datapath, 
                            skiprows=range(1, 417062), 
                            nrows=606462-417062)

# Concatenate the two parts
patient = pd.concat([patient1_part1, patient1_part2, patient1_part3, patient3_part1, patient3_part2, patient3_part3])
therapist = pd.concat([therapist1_part1, therapist1_part2, therapist1_part3, therapist3_part1, therapist3_part2, therapist3_part3])

patient_data = patient[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                        ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4']].values.astype('float32')
therapist_data = therapist[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

patient_valid = patient2_part1[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                                ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4']].values.astype('float32')
therapist_valid = therapist2_part1[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

timeseries = np.column_stack((patient_data, therapist_data))
timeseries_valid = np.column_stack((patient_valid, therapist_valid))
 
# plt.plot(timeseries)
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions')
# plt.title('Left Hip Joint Positions Over Time')
# plt.legend(['Patient', 'Therapist'])
# plt.xlim(0, 7500)
# plt.show()

# train-valid split for time series
train_size = int(len(timeseries))
valid_size = int(len(timeseries_valid * 0.5))
train, valid = timeseries[:train_size], timeseries_valid[:valid_size]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, :8]  # Feature is patient data
        target = dataset[i+1:i+lookback+1, -4:]  # Target is therapist data
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
    

model = JointModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=128)

train_loss_list = []
valid_loss_list = []
n_epochs = 100
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
    model.eval()
    with torch.no_grad():
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
plt.ylabel('Loss (RMSE)')
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

# Plotting
with torch.no_grad():
    # Create arrays for plotting
    therapist_true = np.ones_like(timeseries[:, 1]) * np.nan
    therapist_pred = np.ones_like(timeseries[:, 1]) * np.nan
    
    # Fill in the true therapist data
    therapist_true[lookback:train_size] = timeseries[lookback:train_size, 1]

    # Create arrays for valid portion
    valid_true = np.ones_like(valid[:, 1]) * np.nan
    valid_pred = np.ones_like(valid[:, 1]) * np.nan
    
    # Fill in true valid data
    valid_true[lookback:valid_size] = valid[lookback:valid_size, 1]
    
    # Get predictions for valid
    valid_predictions = model(X_valid)[:, -1, :].numpy().flatten()
    valid_pred[lookback:valid_size] = valid_predictions
