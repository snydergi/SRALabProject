import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

patient_file = '/home/gis/SRALab_Data/Subject1/X2_SRA_A_07-05-2024_10-39-10.csv'
therapist_file = '/home/gis/SRALab_Data/Subject1/X2_SRA_B_07-05-2024_10-41-46.csv'

patient_data_raw = pd.read_csv(patient_file, low_memory=False)
therapist_data_raw = pd.read_csv(therapist_file, low_memory=False)

# Print column names if you cannot remember the names.
# for c in patient_data_raw.columns:
#     print(c)

patient_time = patient_data_raw['time'].values.astype('float32')
therapist_time = therapist_data_raw['time'].values.astype('float32')
patient_data = patient_data_raw[[' JointPositions_1', ' TimeInteractionSubscription']].values.astype('float32')
therapist_data = therapist_data_raw[[' JointPositions_3', ' TimeInteractionSubscription']].values.astype('float32')

therapist_button_data = therapist_data_raw[' greenButton'].values.astype('float32')
patient_button_data = patient_data_raw[' greenButton'].values.astype('float32')

####################### FOR SYNCHING DATA #########################
index_to_match = 165634

# If therapist determines
print(f'Time to match: {therapist_data[index_to_match, 1]}')
for k in range(len(patient_data[:,1])):
    if np.round(patient_data[k, 1], 1) == np.round(therapist_data[index_to_match, 1], 1):
        print(f'Possible index found at {k}, time {patient_data[k, 1]}')

# If patient determines
# print(f'Time to match: {patient_data[index_to_match, 1]}')
# for k in range(len(therapist_data[:,1])):
#     if np.round(therapist_data[k, 1], 1) == np.round(patient_data[index_to_match, 1], 1):
################################################################


################## FOR DETERMINING APPROXIMATE EPISODE STARTS AND ENDS ##################
# FOR THERAPIST DATA
# therapist_indices = []
# for i in range(len(therapist_button_data)):
#     if i > 0:
#         # For start of episodes
#         if therapist_button_data[i] == 1 and therapist_button_data[i-1] == 0:
#             print(f"Button pressed at index {i}, time {therapist_data[i,1]}")
#             therapist_indices.append(i)
        # For when the button is pressed when data collection starts
        # if therapist_button_data[i] == 1:
        #     print(f"Button pressed at index {i}, time {therapist_time[i]}")
        #     break
        # For end of episodes
        # if therapist_button_data[i] == 0 and therapist_button_data[i-1] == 1:
        #     print(f"Button unpressed at index {i}, time {therapist_time[i]}")

# FOR PATIENT DATA
# patient_indices = []
# for j in range(len(patient_button_data)):
#     if j > 0:
#         # For start of episodes
#         if patient_button_data[j] == 1 and patient_button_data[j-1] == 0:
#             print(f"Button pressed at index {j}, time {patient_data[j,1]}")
#             patient_indices.append(j)
#         # For when the button is pressed when data collection starts
#         # if patient_button_data[i] == 1:
#         #     print(f"Button pressed at index {i}, time {patient_time[i]}")
#         #     break
#         # For end of episodes
#         # if patient_button_data[i] == 0 and patient_button_data[i-1] == 1:
#         #     print(f"Button unpressed at index {i}, time {patient_time[i]}")

# fig = plt.figure(figsize=(12, 6))
# plt.plot(therapist_data[:,1], therapist_data[:,0], label='Therapist Joint Positions', c='r')
# plt.plot(patient_data[:,1], patient_data[:,0], label='Patient Joint Positions', c='b')
# plt.plot(therapist_data[:,1], therapist_data_raw[' greenButton'].values.astype('float32'), label='Green Button Value T', c='green')
# plt.plot(patient_data[:,1], patient_data_raw[' greenButton'].values.astype('float32'), label='Green Button Value P', c='yellow')
# plt.legend()
# plt.show()
###############################################################

# print(therapist_data[0, 1])
# print(therapist_data[-1, 1])
# print(len(therapist_data))

# print(patient_data[0, 1])
# print(patient_data[-1, 1])
# print(len(patient_data))