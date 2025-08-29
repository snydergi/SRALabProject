"""Chart inference times from CSV file."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read inference times from CSV file
inference_times = []
skippedFirstRow = True
with open('/home/gis/SRALab_Data/ExperimentDay3/prediction_1753216108.9706633.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if skippedFirstRow:
            skippedFirstRow = False
            continue
        if row[0] == 'trial6':
            inference_times.append(float(row[1]))

inference_times = np.array(inference_times)

fig = plt.figure(figsize=(12,6), )
plt.hist(inference_times, bins=25, alpha=0.7, color='blue')
plt.xlabel('Inference Time (Seconds)')
plt.ylabel('# of Occurrances (Log Scale)')
plt.yscale('log')
plt.title(f'Trial6 Model, Sample of {len(inference_times)} Inference Times, Mean: {np.mean(inference_times):.5f}, Max: {inference_times.max():.5f}, Min: {inference_times.min():.5f}')
plt.show()
