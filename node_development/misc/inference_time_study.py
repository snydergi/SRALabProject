import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read inference times from CSV file
inference_times = []
with open('/home/gis/SRALab_Data/ExperimentDay1/time_inference_1752606480.2458458.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[1] == 'trial6':
            inference_times.append(float(row[2]))

inference_times = np.array(inference_times)

fig = plt.figure(figsize=(12,6), )
plt.hist(inference_times, bins=25, alpha=0.7, color='blue')
plt.xlabel('Inference Time (Seconds)')
plt.ylabel('# of Occurrances (Log Scale)')
plt.yscale('log')
plt.title(f'Trial6 Model, Sample of {len(inference_times)} Inference Times, Mean: {np.mean(inference_times):.5f}, Max: {inference_times.max():.5f}, Min: {inference_times.min():.5f}')
plt.show()