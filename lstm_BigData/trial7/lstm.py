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
patient2_part2 = pd.read_csv(patient2_datapath, 
                     skiprows=range(1, 457108), 
                     nrows=595193-457108)
therapist2_part2 = pd.read_csv(therapist2_datapath, 
                       skiprows=range(1, 457108), 
                       nrows=595193-457108)

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

# Fourth Subject-Therapist Pair
patient4_datapath = '../data/Patient4_X2_SRA_A_31-07-2024_11-25-24.csv'
therapist4_datapath = '../data/Therapist4_X2_SRA_B_31-07-2024_11-29-20.csv'
patient4_part1 = pd.read_csv(patient4_datapath, 
                          skiprows=range(1, 85602), 
                          nrows=285867-85602)
therapist4_part1 = pd.read_csv(therapist4_datapath, 
                            skiprows=range(1, 28003), 
                            nrows=228268-28003)
patient4_part2 = pd.read_csv(patient4_datapath, 
                          skiprows=range(1, 307526), 
                          nrows=435811-307526)
therapist4_part2 = pd.read_csv(therapist4_datapath, 
                            skiprows=range(1, 249927), 
                            nrows=378212-249927)
patient4_part3 = pd.read_csv(patient4_datapath, 
                          skiprows=range(1, 481091), 
                          nrows=593066-481091)
therapist4_part3 = pd.read_csv(therapist4_datapath, 
                            skiprows=range(1, 423491), 
                            nrows=535482-423491)
patient4_part4 = pd.read_csv(patient4_datapath,
                            skiprows=range(1, 610823), 
                            nrows=688030-610823)
therapist4_part4 = pd.read_csv(therapist4_datapath,
                            skiprows=range(1, 553222), 
                            nrows=630428-553222)

# Sixth Subject-Therapist Pair
patient6_datapath = '../data/Patient6_X2_SRA_A_15-08-2024_09-47-25.csv'
therapist6_datapath = '../data/Therapist6_X2_SRA_B_15-08-2024_09-47-47.csv'
patient6_part1 = pd.read_csv(patient6_datapath, 
                          skiprows=range(1, 7695), 
                          nrows=205185-7695)
therapist6_part1 = pd.read_csv(therapist6_datapath, 
                            skiprows=range(1, 7695), 
                            nrows=205185-7695)
patient6_part2 = pd.read_csv(patient6_datapath, 
                          skiprows=range(1, 215454), 
                          nrows=419150-215454)
therapist6_part2 = pd.read_csv(therapist6_datapath, 
                            skiprows=range(1, 215454), 
                            nrows=419150-215454)
patient6_part3 = pd.read_csv(patient6_datapath, 
                          skiprows=range(1, 456313), 
                          nrows=656732-456313)
therapist6_part3 = pd.read_csv(therapist6_datapath, 
                            skiprows=range(1, 456313), 
                            nrows=656732-456313)

# Ninth Subject-Therapist Pair
patient9_datapath = '../data/Patient9_X2_SRA_A_07-10-2024_14-41-04.csv'
therapist9_datapath = '../data/Therapist9_X2_SRA_B_07-10-2024_14-41-12.csv'
patient9_part1 = pd.read_csv(patient9_datapath, 
                          skiprows=range(1, 52347), 
                          nrows=245511-52347)
therapist9_part1 = pd.read_csv(therapist9_datapath, 
                            skiprows=range(1, 46634), 
                            nrows=239798-46634)
patient9_part2 = pd.read_csv(patient9_datapath, 
                          skiprows=range(1, 373504), 
                          nrows=560745-373504)
therapist9_part2 = pd.read_csv(therapist9_datapath, 
                            skiprows=range(1, 367792), 
                            nrows=555032-367792)
patient9_part3 = pd.read_csv(patient9_datapath, 
                          skiprows=range(1, 732658), 
                          nrows=917918-732658)
therapist9_part3 = pd.read_csv(therapist9_datapath, 
                            skiprows=range(1, 726946), 
                            nrows=912205-726946)

# Eleventh Subject-Therapist Pair
patient11_datapath = '../data/Patient11_X2_SRA_A_19-09-2024_09-37-01.csv'
therapist11_datapath = '../data/Therapist11_X2_SRA_B_19-09-2024_09-37-07.csv'
patient11_part1 = pd.read_csv(patient11_datapath, 
                          skiprows=range(1, 36738), 
                          nrows=97920-36738)
therapist11_part1 = pd.read_csv(therapist11_datapath, 
                            skiprows=range(1, 36738), 
                            nrows=97920-36738)
patient11_part2 = pd.read_csv(patient11_datapath, 
                          skiprows=range(1, 106812), 
                          nrows=249479-106812)
therapist11_part2 = pd.read_csv(therapist11_datapath, 
                            skiprows=range(1, 106812), 
                            nrows=249479-106812)
patient11_part3 = pd.read_csv(patient11_datapath, 
                          skiprows=range(1, 261278), 
                          nrows=459644-261278)
therapist11_part3 = pd.read_csv(therapist11_datapath, 
                            skiprows=range(1, 261278), 
                            nrows=459644-261278)
patient11_part4 = pd.read_csv(patient11_datapath,
                            skiprows=range(1, 585102), 
                            nrows=787871-585102)
therapist11_part4 = pd.read_csv(therapist11_datapath,
                            skiprows=range(1, 585102), 
                            nrows=787871-585102)

p1 = pd.concat([patient1_part1, patient1_part2, patient1_part3])
t1 = pd.concat([therapist1_part1, therapist1_part2, therapist1_part3])
p2 = pd.concat([patient2_part1, patient2_part2])
t2 = pd.concat([therapist2_part1, therapist2_part2])
p3 = pd.concat([patient3_part1, patient3_part2, patient3_part3])
t3 = pd.concat([therapist3_part1, therapist3_part2, therapist3_part3])
p4 = pd.concat([patient4_part1, patient4_part2, patient4_part3, patient4_part4])
t4 = pd.concat([therapist4_part1, therapist4_part2, therapist4_part3, therapist4_part4])
p6 = pd.concat([patient6_part1, patient6_part2, patient6_part3])
t6 = pd.concat([therapist6_part1, therapist6_part2, therapist6_part3])
p9 = pd.concat([patient9_part1, patient9_part2, patient9_part3])
t9 = pd.concat([therapist9_part1, therapist9_part2, therapist9_part3])
p11 = pd.concat([patient11_part1, patient11_part2, patient11_part3, patient11_part4])
t11 = pd.concat([therapist11_part1, therapist11_part2, therapist11_part3, therapist11_part4])

train_1_len = len(p1) * 0.7
validate_1_len = len(p1) * 0.2
train_2_len = len(p2) * 0.7
validate_2_len = len(p2) * 0.2
train_3_len = len(p3) * 0.7
validate_3_len = len(p3) * 0.2
train_4_len = len(p4) * 0.7
validate_4_len = len(p4) * 0.2
train_6_len = len(p6) * 0.7
validate_6_len = len(p6) * 0.2
train_9_len = len(p9) * 0.7
validate_9_len = len(p9) * 0.2
train_11_len = len(p11) * 0.7
validate_11_len = len(p11) * 0.2

# Split the data into training and validation sets
patient1_train = p1[:int(train_1_len)]
therapist1_train = t1[:int(train_1_len)]
patient1_valid = p1[int(train_1_len):int(train_1_len + validate_1_len)]
therapist1_valid = t1[int(train_1_len):int(train_1_len + validate_1_len)]

patient2_train = p2[:int(train_2_len)]
therapist2_train = t2[:int(train_2_len)]
patient2_valid = p2[int(train_2_len):int(train_2_len + validate_2_len)]
therapist2_valid = t2[int(train_2_len):int(train_2_len + validate_2_len)]

patient3_train = p3[:int(train_3_len)]
therapist3_train = t3[:int(train_3_len)]
patient3_valid = p3[int(train_3_len):int(train_3_len + validate_3_len)]
therapist3_valid = t3[int(train_3_len):int(train_3_len + validate_3_len)]

patient4_train = p4[:int(train_4_len)]
therapist4_train = t4[:int(train_4_len)]
patient4_valid = p4[int(train_4_len):int(train_4_len + validate_4_len)]
therapist4_valid = t4[int(train_4_len):int(train_4_len + validate_4_len)]

patient6_train = p6[:int(train_6_len)]
therapist6_train = t6[:int(train_6_len)]
patient6_valid = p6[int(train_6_len):int(train_6_len + validate_6_len)]
therapist6_valid = t6[int(train_6_len):int(train_6_len + validate_6_len)]

patient9_train = p9[:int(train_9_len)]
therapist9_train = t9[:int(train_9_len)]
patient9_valid = p9[int(train_9_len):int(train_9_len + validate_9_len)]
therapist9_valid = t9[int(train_9_len):int(train_9_len + validate_9_len)]

patient11_train = p11[:int(train_11_len)]
therapist11_train = t11[:int(train_11_len)]
patient11_valid = p11[int(train_11_len):int(train_11_len + validate_11_len)]
therapist11_valid = t11[int(train_11_len):int(train_11_len + validate_11_len)]

# Concatenate
patient_train = pd.concat([patient1_train, patient2_train, patient3_train, patient4_train, patient6_train, patient9_train, patient11_train])
therapist_train = pd.concat([therapist1_train, therapist2_train, therapist3_train, therapist4_train, therapist6_train, therapist9_train, therapist11_train])
patient_valid = pd.concat([patient1_valid, patient2_valid, patient3_valid, patient4_valid, patient6_valid, patient9_valid, patient11_valid])
therapist_valid = pd.concat([therapist1_valid, therapist2_valid, therapist3_valid, therapist4_valid, therapist6_valid, therapist9_valid, therapist11_valid])

patient_data = patient_train[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                              ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4']].values.astype('float32')
therapist_data = therapist_train[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

patient_valid = patient_valid[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                               ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4']].values.astype('float32')
therapist_valid = therapist_valid[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

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
        self.lstm = nn.LSTM(input_size=8, hidden_size=50, num_layers=1, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(50, 4)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    

model = JointModel()
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
