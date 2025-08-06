from bagpy import bagreader

bag = bagreader('/home/cerebro/catkin_ws/2025-08-06-15-42-49.bag') # Path to rosbag
msg = bag.message_by_topic('/X2_SRA_B/custom_robot_state') # Select topic for conversion to csv

print(msg) # Prints path of saved csv