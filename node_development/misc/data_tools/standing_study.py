import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read inference times from CSV file
j1 = []
j2 = []
j3 = []
j4 = []
with open('/home/cerebro/snyder_project/SRALabProject/node_development/misc/pred_not_walking.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        j1.append(float(row[0]))
        j2.append(float(row[1]))
        j3.append(float(row[2]))
        j4.append(float(row[3]))

j1 = np.array(j1)
j2 = np.array(j2)
j3 = np.array(j3)
j4 = np.array(j4)

fig = plt.figure(figsize=(12,6), )
fig.suptitle('Predicted Joint Angles for Not Walking (No Wearer, Hanging on Stand)', fontsize=16)
plt.subplot(2,2,1)
plt.title('Joint 1')
plt.ylabel('Angle (rad)')
plt.ylim([-0.5, 1.5])
plt.xlabel('Time Step')
plt.plot(j1, label='Joint 1')
plt.subplot(2,2,2)
plt.title('Joint 2')
plt.ylabel('Angle (rad)')
plt.ylim([-2.0, 0.0])
plt.xlabel('Time Step')
plt.plot(j2, label='Joint 2')
plt.subplot(2,2,3)
plt.title('Joint 3')
plt.ylabel('Angle (rad)')
plt.ylim([-0.5, 1.5])
plt.xlabel('Time Step')
plt.plot(j3, label='Joint 3')
plt.subplot(2,2,4)
plt.title('Joint 4')
plt.ylabel('Angle (rad)')
plt.ylim([-2.0, 0.0])
plt.xlabel('Time Step')
plt.plot(j4, label='Joint 4')
plt.show()