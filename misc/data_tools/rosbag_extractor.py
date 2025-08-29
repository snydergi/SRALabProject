"""Extracts a specific topic from a ROS bag file and saves it as a CSV file."""

from bagpy import bagreader

bag = bagreader('/home/gis/SRALab_Data/StackedTestDay1/2025-08-14-16-22-37.bag') # Path to rosbag
msg = bag.message_by_topic('/X2_SRA_A/joint_states/velocity') # Select topic for conversion to csv

print(msg) # Prints path of saved csv
