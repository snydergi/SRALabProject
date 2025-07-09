import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read inference times from CSV file
inference_times = []
with open('/home/cerebro/snyder_project/SRALabProject/node_development/misc/inference_time.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        inference_times.append(float(row[0]))

inference_times = np.array(inference_times)

fig = plt.figure(figsize=(12,6), )
plt.hist(inference_times, bins=25, alpha=0.7, color='blue')
plt.xlabel('Inference Time (Seconds)')
plt.ylabel('# of Occurrances (Log Scale)')
# plt.yscale('log')
plt.title(f'Sample of {len(inference_times)} Inference Times, Mean: {np.mean(inference_times)}, Max: {inference_times.max()}, Min: {inference_times.min()}')
plt.show()