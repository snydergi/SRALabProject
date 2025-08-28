# Synthetic Therapist Setup
The Synthetic Therapist has both hardware and software components necessary for function. 

For hardware, this project utilizes a modified ExoMotus X2 (Fourier Intelligence) lower-limb exoskeleton. The device is actuated at the hips and knees, with ankle joints being capable of passive motion. Communication with the exoskeleton system via the CANOpen communication control protocol over CAN bus. For exoskeleton setup, contact [Lorenzo Vianello](https://github.com/LorenzoVianello95) for access to the document [here](https://github.com/ponsLab/BACKUP_CANOpenRobotController/blob/master/doc/1.GettingStarted/GettingStarted.md).

Software for the Synthetic Therapist is based on ROS. Exoskeleton controllers are developed on the CANOpen Robot Controller (CORC). Further information about CORC setup can be found [here](https://github.com/ponsLab/BACKUP_CANOpenRobotController/blob/master/doc/1.GettingStarted/GettingStarted.md). For specifics on which branch to use, contact [Lorenzo Vianello](https://github.com/LorenzoVianello95).

The 'synthetic_therapist' package found in this repository should be downloaded and placed in the catkin workspace ```src``` folder alongside packages described in the CORC setup.

Use ```catkin build``` to build the workspace. Be sure to run ```source devel/setup.bash``` in each new terminal window.

Key commands to run are as follows:
- ```roscore``` for basic ROS function
- ```roslaunch CORC x2_real_A.launch``` to run the launch file for the exoskeleton
- ```rqt``` to have access to further exoskeleton setup and dynamic reconfigure variables
- ```rqt_multiplot``` for visualization of input data and predictions
- ```rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0 _baud:=115200``` to initialize the treadmill

Launch the Synthetic Therapist node with ```rosrun synthetic_therapist model_node.py```. The default model selected for the Synthetic Therapist uses patient hip and knee joint positions and velocities to predict therapist hip and knee joint positions and velocities between ~3ms to ~75ms in the future. The distance into the future for prediction can be tuned with dynamic reconfigure in ```rqt``` by adjusting the ```future_distance``` parameter. Additional parameters that may be changed there are ```model_type``` and ```pred_diff_threshold```, which change the active LSTM and the joint limits for predictions respectively. 

# Model Training
## Virtual Environment
Use ```requirements.txt``` for creating your virtual environment.

## Dataset
To utilize this repository, first clone it into a fresh folder. The code requires users to have access to patient and therapist gait data in the form of CSVs containing L/R hip joint positions and velocities, L/R knee joint positions and velocities, backpack position and velocity, and timesteps for synchronization. Datasets will not be included in this respository as, though de-identified from any patient information, they are best kept private.

## Models
Subfolders in this directory represent milestones in the synthetic therapist development. [```lstm_BigData```](lstm_BigData/README.md) contains code used to generate the most up to date iterations of models, and all [five models](synthetic_therapist/models/) that are selectable within the synthetic therapist. 

See each folder's `README.md` for additional information about model structure.

# About
This repository contains code written for the Shirley Ryan AbilityLab Pons Lab. 

The goal of the project is to further developments on patient-therapist interactions involving use of lower-body exoskeletons through application of machine-learning models by creating a 'Synthetic Therapist'.

This code is authored by Grayson Snyder as part of his final project for the MS in Robotics degree from Northwestern University.
