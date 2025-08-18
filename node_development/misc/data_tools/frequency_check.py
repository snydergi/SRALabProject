import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

# For full data (from preliminary data collection)
times = []
skippedFirstRow = True
filepath = '/home/gis/SRALab_Data/Patient11_X2_SRA_A_19-09-2024_09-37-01.csv'
with open(filepath, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if skippedFirstRow:
            print(row[171])
            skippedFirstRow = False
            continue
        times.append(float(row[171])) # Time Data Column. (TimeInteractionSubscription)
times = np.array(times)
dts = []
for i in range(len(times) - 1):
    if times[i + 1] - times[i] > 0 and times[i + 1] - times[i] < 1: # Filter out dt of 0 or large jumps
        dts.append(times[i + 1] - times[i])
dts = np.array(dts)
print(f'Frequency: {1 / np.mean(dts)} Hz')
print(f'Standard Deviation: {np.std(dts)}')
print(f'Average Time Step: {np.mean(dts)} seconds')