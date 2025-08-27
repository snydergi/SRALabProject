import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

joint_predictions = []
with open('/home/gis/SRALab_Data/ExperimentDay3/prediction_1753215345.0074022.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == 'trial4':
            joint_predictions.append(float(row[2]))

joint_predictions = np.array(joint_predictions)

fig = plt.figure(figsize=(12,6), )
# plt.plot(joint_predictions, alpha=0.7, color='blue')
plt.scatter(np.arange(len(joint_predictions)), joint_predictions, alpha=0.7, color='blue', s=1)
plt.xlabel('Time Step')
plt.ylabel('Radians')
# plt.xlim(185000,205000)
plt.title(f'Joint 1 Predictions')
plt.show()