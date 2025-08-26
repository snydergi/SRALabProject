# About
This repository contains code written for the Shirley Ryan AbilityLab Pons Lab. 

The goal of the project is to further developments on patient-therapist interactions involving use of lower-body exoskeletons through application of machine-learning models by creating a 'Synthetic Therapist'.

This code is authored by Grayson Snyder as part of his final project for the MS in Robotics degree from Northwestern University.

# Quickstart
## Dataset
To utilize this repository, first clone it into a fresh folder. The code requires users to have access to patient and therapist gait data in the form of CSVs containing L/R hip joint positions and velocities, L/R knee joint positions and velocities, backpack position and velocity, and timesteps for synchronization. Datasets will not be included in this respository as, though de-identified from any patient information, they are best kept private.

## Virtual Environment
Create a virtual environment in the root directory of the repository with `python3 -m venv venv` to create a virtual environment called `venv`. Next, run the command `source venv/bin/activate` to activate the virtual environment you just created. At this point, you will need to install dependencies with `pip`. These include: `torch`, `numpy`, `matplotlib`, and `pandas`. If there are additional dependencies requested by the code when you run it later, download those as well.

## Synthetic Therapist
The Synthetic Therapist has both hardware and software components necessary for function. 

For hardware, this project utilizes a modified ExoMotus X2 (Fourier Intelligence) lower-limb exoskeleton. The device is actuated at the hips and knees, with ankle joints being capable of passive motion. Communication with the exoskeleton system via the CANOpen communication control protocol over CAN bus. 

Software for the Synthetic Therapist is based on ROS1 Noetic. Exoskeleton controllers are developed on the CANOpen Robot Controller (CORC). Further information about CORC can be found [here](https://github.com/UniMelbHumanRoboticsLab/CANOpenRobotController).

Contributions to the Synthetic Therapist found in this repository are trained models and a ROS1 Noetic node created for integration into the Pons Lab's pre-existing (proprietary) workspace.


See each folder's `README.md` for additional instructions for running the various models.