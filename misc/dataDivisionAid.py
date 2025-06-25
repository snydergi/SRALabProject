import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

patient_file = '/home/gis/SRALab_Data/Subject3/X2_SRA_A_29-05-2024_13-36-40.csv'
therapist_file = '/home/gis/SRALab_Data/Subject3/X2_SRA_B_29-05-2024_13-41-19.csv'

patient_data_raw = pd.read_csv(patient_file, low_memory=False)
therapist_data_raw = pd.read_csv(therapist_file, low_memory=False)

# for c in patient_data_raw.columns:
#     print(c)

patient_time = patient_data_raw['time'].values.astype('float32')
therapist_time = therapist_data_raw['time'].values.astype('float32')
patient_data = patient_data_raw[[' JointPositions_1', ' JointPositions_3']].values.astype('float32')
therapist_data = therapist_data_raw[[' JointPositions_1', ' JointPositions_3']].values.astype('float32')

therapist_button_data = therapist_data_raw[' greenButton'].values.astype('float32')
patient_button_data = patient_data_raw[' greenButton'].values.astype('float32')

for i in range(len(therapist_button_data)):
    if i > 0:
        # For start of episodes
        # if therapist_button_data[i] == 1 and therapist_button_data[i-1] == 0:
        #     print(f"Button pressed at index {i}, time {therapist_time[i]}")
        # For when the button is pressed when data collection starts
        # if therapist_button_data[i] == 1:
        #     print(f"Button pressed at index {i}, time {therapist_time[i]}")
        #     break
        # For end of episodes
        if therapist_button_data[i] == 0 and therapist_button_data[i-1] == 1:
            print(f"Button unpressed at index {i}, time {therapist_time[i]}")

# print(f'len patient data: {len(patient_data)}')
# print(f'len patient time: {len(patient_time)}')
# print(f'patient time at 0: {patient_time[0]}')

# for i in range(len(patient_button_data)):
#     if i > 0:
#         # For start of episodes
#         # if patient_button_data[i] == 1 and patient_button_data[i-1] == 0:
#         #     print(f"Button pressed at index {i}, time {patient_time[i]}")
#         # For when the button is pressed when data collection starts
#         # if patient_button_data[i] == 1:
#         #     print(f"Button pressed at index {i}, time {patient_time[i]}")
#         #     break
#         # For end of episodes
#         # if patient_button_data[i] == 0 and patient_button_data[i-1] == 1:
#         #     print(f"Button unpressed at index {i}, time {patient_time[i]}")
        

fig = plt.figure(figsize=(12, 6))
plt.plot(therapist_time, therapist_data[:,0], label='Therapist Joint Positions', c='r')
# plt.plot(patient_time, patient_data[:,0], label='Patient Joint Positions', c='b')
plt.plot(therapist_time, therapist_data_raw[' greenButton'].values.astype('float32'), label='Green Button Value T', c='green')
# plt.plot(patient_time, patient_data_raw[' greenButton'].values.astype('float32'), label='Green Button Value P', c='yellow')
plt.show()
