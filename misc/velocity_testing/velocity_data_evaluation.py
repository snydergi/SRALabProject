"""Attempt to use a Butterworth low-pass filter to clean up velocity data from CSV file."""

from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = '/home/cerebro/catkin_ws/2025-08-06-15-42-49/X2_SRA_B-custom_robot_state.csv'
raw = pd.read_csv(csv_path,
                  skiprows=range(1,11500), # Skip lines from preparation
                  nrows=35000-11500) # Trim some off of end 

full_data = raw['joint_state.velocity'] # Each row contains five velocities

def parse_row(row):
    """Parse a row of joint velocities from the CSV file.
    Args:
        row (str): A string representing a row of joint velocities in the format '[v1, v2, v3, v4, v5]'.
    Returns:
        list: A list of floats representing the joint velocities.
    """
    return [float(x.strip()) for x in row[1:-1].split(',')]

parsed_data = full_data.apply(parse_row)

jt_vel = np.array(parsed_data.tolist())
jt1_vel = jt_vel[:, 0]
jt2_vel = jt_vel[:, 1]
jt3_vel = jt_vel[:, 2]
jt4_vel = jt_vel[:, 3]

fs = 333 # sampling frequency
dt = 1.0 / fs
N = len(jt1_vel)
t = np.linspace(0, N * dt, N)

# Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    """Design a Butterworth low-pass filter.
    Args:
        cutoff (float): Cutoff frequency of the filter in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter. Default is 5.
    Returns:
        tuple: Numerator (b) and denominator (a) of the filter's transfer function.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a # Numerator (b) and denominator (a) of polunomials of the filter

# Apply filter to data
def apply_filter(data, cutoff, fs, order=5):
    """Apply a Butterworth low-pass filter to the data.
    Args:
        data (np.ndarray): Input signal data.
        cutoff (float): Cutoff frequency of the filter in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter. Default is 5.
    Returns:
        np.ndarray: Filtered signal data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)  # Zero-phase filtering (no lag)
    return y

# Apply filter to each joint
cutoff_freq = 1.0  # Adjust based on your signal characteristics
jt1_vel_filtered = apply_filter(jt1_vel, cutoff_freq, fs)
jt2_vel_filtered = apply_filter(jt2_vel, cutoff_freq, fs)
jt3_vel_filtered = apply_filter(jt3_vel, cutoff_freq, fs)
jt4_vel_filtered = apply_filter(jt4_vel, cutoff_freq, fs)

# Plot original vs filtered data
plt.figure(figsize=(10, 6))
plt.plot(t, jt1_vel, 'b-', alpha=0.5, label='Joint 1 Raw')
plt.plot(t, jt1_vel_filtered, 'r-', linewidth=1.5, label='Joint 1 Filtered')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (rad/s)')
plt.legend()
plt.title('Joint 1 Velocity: Raw vs Filtered')
plt.show()
