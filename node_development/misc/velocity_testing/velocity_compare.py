import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = '/home/gis/SRALab_Data/StackedTestDay1/velocities_1755205889.4338095.csv'
raw = pd.read_csv(csv_path, skiprows=range(0, 68000), nrows=72000-68000)

# # Columns in order: Time, Pred Joint 1, Pred Joint 2, Pred Joint 3, Pred Joint 4,
# #                   Diff Joint 1, Diff Joint 2, Diff Joint 3, Diff Joint 4
# # For pred/diff data
desired_joint = 1
pred_col = 1 if desired_joint == 1 else 2 if desired_joint == 2 else 3 if desired_joint == 3 else 4
diff_col = 5 if desired_joint == 1 else 6 if desired_joint == 2 else 7 if desired_joint == 3 else 8

pred_vels = raw.iloc[:, pred_col].values
diff_vels = raw.iloc[:, diff_col].values

fig = plt.figure(figsize=(12, 6))
plt.plot(pred_vels, label=f'Predicted Joint {desired_joint} Velocity', color='blue', ls=':')
plt.plot(diff_vels, label=f'Numerical Differentiation Joint {desired_joint} Velocity', color='red', ls='-', linewidth=0.5, alpha=0.5)
plt.xlim(0, 1500)
plt.xlabel('Time Step (~3ms)')
plt.ylabel('Velocity (rad/s)')
plt.title(f'Joint {desired_joint} Velocity Comparison')
plt.legend()
# plt.show()
plt.savefig(f'joint_{desired_joint}_velocity_comparison.svg', dpi=300, bbox_inches='tight', format='svg')
