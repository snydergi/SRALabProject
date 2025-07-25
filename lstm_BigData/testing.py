import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from torchrl.modules import NoisyLinear

# First Subject-Therapist Pair
patient1_datapath = 'data/Patient1_X2_SRA_A_07-05-2024_10-39-10.csv'
therapist1_datapath = 'data/Therapist1_X2_SRA_B_07-05-2024_10-41-46.csv'
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
patient2_datapath = 'data/Patient2_X2_SRA_A_08-05-2024_14-33-44.csv'
therapist2_datapath = 'data/Therapist2_X2_SRA_B_08-05-2024_14-33-51.csv'
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
patient3_datapath = 'data/Patient3_X2_SRA_A_29-05-2024_13-36-40.csv'
therapist3_datapath = 'data/Therapist3_X2_SRA_B_29-05-2024_13-41-19.csv'
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

p1 = pd.concat([patient1_part1, patient1_part2, patient1_part3])
t1 = pd.concat([therapist1_part1, therapist1_part2, therapist1_part3])
p2 = pd.concat([patient2_part1, patient2_part2])
t2 = pd.concat([therapist2_part1, therapist2_part2])
p3 = pd.concat([patient3_part1, patient3_part2, patient3_part3])
t3 = pd.concat([therapist3_part1, therapist3_part2, therapist3_part3])

train_1_len = len(p1) * 0.7
validate_1_len = len(p1) * 0.2
train_2_len = len(p2) * 0.7
validate_2_len = len(p2) * 0.2
train_3_len = len(p3) * 0.7
validate_3_len = len(p3) * 0.2

# Split testing portion of data from full dataset
patient1_test = p1[int(train_1_len + validate_1_len):]
therapist1_test = t1[int(train_1_len + validate_1_len):]

patient2_test = p2[int(train_2_len + validate_2_len):]
therapist2_test = t2[int(train_2_len + validate_2_len):]

patient3_test = p3[int(train_3_len + validate_3_len):]
therapist3_test = t3[int(train_3_len + validate_3_len):]

patient1_data = patient1_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                              ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4',
                              ' StanceInterpolationFactor', ' BackPackAngle', ' BackPackAngularVelocity']].values.astype('float32')
therapist1_data = therapist1_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

patient2_data = patient2_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                               ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4',
                               ' StanceInterpolationFactor', ' BackPackAngle', ' BackPackAngularVelocity']].values.astype('float32')
therapist2_data = therapist2_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

patient3_data = patient3_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                               ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4',
                               ' StanceInterpolationFactor', ' BackPackAngle', ' BackPackAngularVelocity']].values.astype('float32')
therapist3_data = therapist3_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

test_1 = np.column_stack((patient1_data, therapist1_data))
test_2 = np.column_stack((patient2_data, therapist2_data))
test_3 = np.column_stack((patient3_data, therapist3_data))


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series data
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, :11]  # Feature is patient data, if input size 8
        # feature = dataset[i:i+lookback, :4]  # Feature is patient data, # if input size 4
        target = dataset[i+1:i+lookback+1, -4:]  # Target is therapist data
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)

lookback = 50
patient_number = 1 # Set based on which patient data is being tested
test = test_1 if patient_number == 1 else test_2 if patient_number == 2 else test_3
X_test, y_test = create_dataset(test, lookback=lookback)

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
model.load_state_dict(torch.load('trial4/lstm_model_epoch198.pth'))
# model = torch.jit.load('/home/cerebro/snyder_project/SRALabProject/misc/model_scripting/scripts/lstm_trial5.pt')
model.eval()

# Testing
with torch.no_grad():
    y_pred = model(X_test)
    loss_fn = nn.MSELoss()
    test_rmse = torch.sqrt(loss_fn(y_pred, y_test))
    print(f"Testing RMSE: {test_rmse:.4f}")
    joint_rmses = []
    for joint_idx in range(4):
        joint_rmse = torch.sqrt(loss_fn(y_pred[:, :, joint_idx], y_test[:, :, joint_idx]))
        joint_rmses.append(joint_rmse.item())

with torch.no_grad():
    # Get all predictions (taking last timestep from each sequence)
    y_pred2 = model(X_test)[:, -1, :].squeeze()  # Shape: [n_samples]
    y_true = y_test[:, -1, :].squeeze()        # Shape: [n_samples]
    
    # # Pointwise absolute errors
    # errors = torch.abs(y_pred2 - y_true).numpy()  # Convert to numpy for plotting
    # Pointwise errors (signed)
    errors = (y_pred2 - y_true).numpy()  # Convert to numpy for plotting
    print(f'Shape of Errors: {errors.shape}')

# Plotting Prep
with torch.no_grad():
    # Create plotting arrays
    plot_length = len(test)  # Only plot testing portion
    
    # Fill true values (offset by lookback)
    therapist_true = test[:, -4:]
    
    # Get predictions (last timestep of each sequence)
    test_pred = model(X_test)[:, -1, :].numpy()

    # Fill predictions (aligned with true values)
    therapist_pred = np.full_like(therapist_true, np.nan)
    therapist_pred[lookback:] = test_pred

# Overlay of periodic data
# Find peaks in the data and divide into periodic segments
data_peaks0, _ = find_peaks(therapist_true[:,0], height=0.4, distance=1000)
periodic_data0 = [therapist_true[data_peaks0[i]:data_peaks0[i+1],0] for i in range(len(data_peaks0)-1)]
pred_peaks0, _ = find_peaks(therapist_pred[:,0], height=0.4, distance=1000)
periodic_pred0 = [therapist_pred[pred_peaks0[i]:pred_peaks0[i+1],0] for i in range(len(pred_peaks0)-1)]

data_peaks1, _ = find_peaks(-therapist_true[:,1], height=0.4, distance=1000)
periodic_data1 = [therapist_true[data_peaks1[i]:data_peaks1[i+1],1] for i in range(len(data_peaks1)-1)]
pred_peaks1, _ = find_peaks(-therapist_pred[:,1], height=0.3, distance=1000)
periodic_pred1 = [therapist_pred[pred_peaks1[i]:pred_peaks1[i+1],1] for i in range(len(pred_peaks1)-1)]

data_peaks2, _ = find_peaks(therapist_true[:,2], height=0.4, distance=1000)
periodic_data2 = [therapist_true[data_peaks2[i]:data_peaks2[i+1],2] for i in range(len(data_peaks2)-1)]
pred_peaks2, _ = find_peaks(therapist_pred[:,2], height=0.4, distance=1000)
periodic_pred2 = [therapist_pred[pred_peaks2[i]:pred_peaks2[i+1],2] for i in range(len(pred_peaks2)-1)]

data_peaks3, _ = find_peaks(-therapist_true[:,3], height=0.4, distance=1000)
periodic_data3 = [therapist_true[data_peaks3[i]:data_peaks3[i+1],3] for i in range(len(data_peaks3)-1)]
pred_peaks3, _ = find_peaks(-therapist_pred[:,3], height=0.3, distance=1000)
periodic_pred3 = [therapist_pred[pred_peaks3[i]:pred_peaks3[i+1],3] for i in range(len(pred_peaks3)-1)]

# Normalize periodic data
normalized_length = 101
normalized_periodic_data0 = []
normalized_periodic_pred0 = []
for period in periodic_data0:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_data0.append(norm_period)
for period in periodic_pred0:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_pred0.append(norm_period)

normalized_periodic_data1 = []
normalized_periodic_pred1 = []
for period in periodic_data1:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_data1.append(norm_period)
for period in periodic_pred1:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_pred1.append(norm_period)

normalized_periodic_data2 = []
normalized_periodic_pred2 = []
for period in periodic_data2:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_data2.append(norm_period)
for period in periodic_pred2:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_pred2.append(norm_period)

normalized_periodic_data3 = []
normalized_periodic_pred3 = []
for period in periodic_data3:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_data3.append(norm_period)
for period in periodic_pred3:
    cur_time = np.linspace(0, 1, len(period))
    new_time = np.linspace(0, 1, normalized_length)
    interp = interp1d(cur_time, period, kind='linear')
    norm_period = interp(new_time)
    normalized_periodic_pred3.append(norm_period)

# # Get mean and std dev of normalized data and pred
# stacked_data0 = np.vstack(normalized_periodic_data0)
# stacked_pred0 = np.vstack(normalized_periodic_pred0)
# mean_data0 = np.mean(stacked_data0, axis=0)
# mean_pred0 = np.mean(stacked_pred0, axis=0)
# std_data0 = np.std(stacked_data0, axis=0)
# std_pred0 = np.std(stacked_pred0, axis=0)
# print(f'Stacked Data0 Size: {stacked_data0.shape}')
# print(f'Stacked Pred0 Size: {stacked_pred0.shape}')

# stacked_data1 = np.vstack(normalized_periodic_data1)
# stacked_pred1 = np.vstack(normalized_periodic_pred1)
# mean_data1 = np.mean(stacked_data1, axis=0)
# mean_pred1 = np.mean(stacked_pred1, axis=0)
# std_data1 = np.std(stacked_data1, axis=0)
# std_pred1 = np.std(stacked_pred1, axis=0)
# print(f'Stacked Data1 Size: {stacked_data1.shape}')
# print(f'Stacked Pred1 Size: {stacked_pred1.shape}')

# stacked_data2 = np.vstack(normalized_periodic_data2)
# stacked_pred2 = np.vstack(normalized_periodic_pred2)
# mean_data2 = np.mean(stacked_data2, axis=0)
# mean_pred2 = np.mean(stacked_pred2, axis=0)
# std_data2 = np.std(stacked_data2, axis=0)
# std_pred2 = np.std(stacked_pred2, axis=0)
# print(f'Stacked Data2 Size: {stacked_data2.shape}')
# print(f'Stacked Pred2 Size: {stacked_pred2.shape}')

# stacked_data3 = np.vstack(normalized_periodic_data3)
# stacked_pred3 = np.vstack(normalized_periodic_pred3)
# mean_data3 = np.mean(stacked_data3, axis=0)
# mean_pred3 = np.mean(stacked_pred3, axis=0)
# std_data3 = np.std(stacked_data3, axis=0)
# std_pred3 = np.std(stacked_pred3, axis=0)
# print(f'Stacked Data3 Size: {stacked_data3.shape}')
# print(f'Stacked Pred3 Size: {stacked_pred3.shape}')

# # # Get errors for periodic data
# # # Must stay commented out until periodic data is set correctly
# mean_err0 = np.mean(abs(stacked_data0 - stacked_pred0), axis=0)
# std_err0 = np.std(abs(stacked_data0 - stacked_pred0), axis=0)

# mean_err1 = np.mean(abs(stacked_data1 - stacked_pred1), axis=0)
# std_err1 = np.std(abs(stacked_data1 - stacked_pred1), axis=0)

# mean_err2 = np.mean(abs(stacked_data2 - stacked_pred2), axis=0)
# std_err2 = np.std(abs(stacked_data2 - stacked_pred2), axis=0)

# mean_err3 = np.mean(abs(stacked_data3 - stacked_pred3), axis=0)
# std_err3 = np.std(abs(stacked_data3 - stacked_pred3), axis=0)

# ALL PLOTTING HAPPENS BELOW HERE. READ INSTRUCTIONS FOR CREATING BEST PLOTS
# Plot histogram of errors
# fig = plt.figure(figsize=(12, 6), )
# fig.suptitle(f"Patient {patient_number} Joint Prediction Error Histograms")
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.hist(errors[:, i], bins=50, alpha=0.7, color='blue')
#     plt.xlabel('Error (Radians)')
#     plt.ylabel('# of Occurrences')
#     plt.title(f'Joint {i+1}, RMSE: {joint_rmses[i]:.4f}, Max (abs): {abs(errors[:,i]).max():.4f}, Std Dev (abs): {abs(errors[:,i]).std():.4f}, Mean (abs): {abs(errors[:,i]).mean():.4f}')
#     plt.grid(True)
# plt.show()

# use to determining amplitude for periodic plots
# fig = plt.figure(figsize=(12, 6))
# joint_pairs = [[0, 2], [1, 3], [2, 0], [3, 1]]  # Pairs of joints to plot together
# plt.plot(therapist_true[:, 1], c='b', label=f'True Therapist Data')
# plt.plot(therapist_pred[:, 1], c='r', linestyle='--', label=f'Predicted Therapist Data')
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions (Radians)')
# plt.legend()
# plt.show()

# Plot only testing data with prediction overlay
# Change xlim for desired time steps
# Limit to 7500 time steps (~30 seconds) EXCEPT when determining amplitude for periodic plots
fig = plt.figure(figsize=(12, 6))
fig.suptitle(f"Patient {patient_number} Testing: Therapist Predictions from Patient Data")
for i in range(4):
    plt.subplot(2, 2, i+1)
    pNum = 2 if i == 0 else 0 if i == 2 else 1 if i == 3 else 3
    plt.plot(test[:, pNum], c='g', label=f'Patient Data')
    plt.plot(therapist_true[:, i], c='b', label=f'True Therapist Data')
    plt.plot(therapist_pred[:, i], c='r', linestyle='--', label=f'Predicted Therapist Data')
    # plt.scatter(np.arange(len(therapist_pred)), therapist_pred[:, i], c='r', marker='x', label=f'Predicted Therapist Data', alpha=0.5)
    plt.xlim(0, 7500)
    plt.xlabel('Time Steps (~4ms)')
    plt.ylabel('Joint Positions (Radians)')
    plt.legend()
    plt.title(f"Joint {i + 1}")
plt.show()
# # plt.savefig('lstm_therapist_prediction.png', dpi=300, bbox_inches='tight')

# # Plot error over time
# plt.figure(figsize=(12, 6))
# plt.plot(errors, c='r', label='Prediction Error')
# plt.xlim(0, 7500)
# plt.xlabel('Time Steps (~4ms)')
# plt.ylabel('Joint Positions Errors (Radians)')
# plt.legend()
# plt.title("Error over Time")
# plt.show()

# Plot normalized periodic data
fig = plt.figure(figsize=(12, 6))

# Plots each period as a separate line. Use to look for outliers in amplitude
# Missteps can cause outliers where multiple periods will be combined into one. Need to fix height in 'find peaks' above
# for period in normalized_periodic_data1:
#     plt.plot(period, c='b', alpha=0.5)
# for period in normalized_periodic_pred1:
#     plt.plot(period, c='r', alpha=0.5)

# Plot mean and std dev of periodic data and predictions
# Use after height is set correctly in 'find peaks' above
# fig.suptitle(f"Patient {patient_number} Joint Periodic Data and Predictions")
# plt.subplot(2, 2, 1)
# plt.plot(mean_data0, c='b', label='Mean Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_data0 - std_data0, 
#                  mean_data0 + std_data0, 
#                  color='b', alpha=0.2)
# plt.plot(mean_pred0, c='r', label='Mean Predicted Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_pred0 - std_pred0, 
#                  mean_pred0 + std_pred0, 
#                  color='r', alpha=0.2)
# plt.title("Left Hip")
# plt.ylabel('Joint Positions (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()
# plt.subplot(2, 2, 2)
# plt.plot(mean_data1, c='b', label='Mean Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_data1 - std_data1, 
#                  mean_data1 + std_data1, 
#                  color='b', alpha=0.2)
# plt.plot(mean_pred1, c='r', label='Mean Predicted Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_pred1 - std_pred1, 
#                  mean_pred1 + std_pred1, 
#                  color='r', alpha=0.2)
# plt.title("Left Knee")
# plt.ylabel('Joint Positions (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.plot(mean_data2, c='b', label='Mean Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_data2 - std_data2, 
#                  mean_data2 + std_data2, 
#                  color='b', alpha=0.2)
# plt.plot(mean_pred2, c='r', label='Mean Predicted Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_pred2 - std_pred2, 
#                  mean_pred2 + std_pred2, 
#                  color='r', alpha=0.2)
# plt.title("Right Hip")
# plt.ylabel('Joint Positions (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()
# plt.subplot(2, 2, 4)
# plt.plot(mean_data3, c='b', label='Mean Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_data3 - std_data3, 
#                  mean_data3 + std_data3, 
#                  color='b', alpha=0.2)
# plt.plot(mean_pred3, c='r', label='Mean Predicted Therapist Data')
# plt.fill_between(range(normalized_length), 
#                  mean_pred3 - std_pred3, 
#                  mean_pred3 + std_pred3, 
#                  color='r', alpha=0.2)
# plt.title("Right Knee")
# plt.ylabel('Joint Positions (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()

# Plot period plot of error using mean and std dev
# plt.suptitle("Joint Periodic Absolute Error")
# plt.subplot(2, 2, 1)
# plt.plot(mean_err0, c='r', label='Mean Error')
# plt.fill_between(range(normalized_length), 
#                  mean_err0 - std_err0, 
#                  mean_err0 + std_err0, 
#                  color='r', alpha=0.2)
# plt.title("Left Hip")
# plt.ylabel('Joint Error (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()
# plt.subplot(2, 2, 2)
# plt.plot(mean_err1, c='r', label='Mean Error')
# plt.fill_between(range(normalized_length), 
#                  mean_err1 - std_err1, 
#                  mean_err1 + std_err1, 
#                  color='r', alpha=0.2)
# plt.title("Left Knee")
# plt.ylabel('Joint Error (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.plot(mean_err2, c='r', label='Mean Error')
# plt.fill_between(range(normalized_length), 
#                  mean_err2 - std_err2, 
#                  mean_err2 + std_err2, 
#                  color='r', alpha=0.2)
# plt.title("Right Hip")
# plt.ylabel('Joint Error (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()
# plt.subplot(2, 2, 4)
# plt.plot(mean_err3, c='r', label='Mean Error')
# plt.fill_between(range(normalized_length), 
#                  mean_err3 - std_err3, 
#                  mean_err3 + std_err3, 
#                  color='r', alpha=0.2)
# plt.title("Right Knee")
# plt.ylabel('Joint Error (Radians)')
# plt.xlabel('Gait Phase %')
# plt.legend()

# plt.show()