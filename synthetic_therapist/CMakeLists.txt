cmake_minimum_required(VERSION 3.0.2)
project(synthetic_therapist)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  dynamic_reconfigure
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_filters
)

generate_dynamic_reconfigure_options(
  cfg/model_node.cfg
)

add_message_files(
  FILES
  X2StanceInterpolation.msg
  X2RobotState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    message_runtime 
    dynamic_reconfigure 
    roscpp 
    rospy 
    std_msgs 
    sensor_msgs
    message_filters
)

catkin_install_python(PROGRAMS
  scripts/model_node.py
  scripts/RingBuffer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg
)
