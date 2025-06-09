# About
This repository contains code written for the Shirley Ryan AbilityLab Pons Group. 

The goal of the project is to further developments on patient-therapist interactions involving use of lower-body exoskeletons through application of machine-learning models.

This code is authored by Grayson Snyder as part of his final project for the MS in Robotics degree from Northwestern University.

# Quickstart
## Dataset
To utilize this repository, first clone it into a fresh folder. The code requires users to have access to patient and therapist gait data in the form of CSVs with columns labeled `TimeInteractionSubscription, JointPositions_1, JointPositions_2, JointPositions_3, JointPositions_4` for models in the folders `lstm_4to1`, `lstm_4to4`, and `lstm_singleJoint`. Datasets will not be included in this respository as, though de-identified from any patient information, they are best kept private.

## Virtual Environment
Currently, one virtual environment is suitable for running any model present in this repository. Create a virtual environment in the root directory of the repository with `python3 -m venv venv` to create a virtual environment called `venv`. Next, run the command `source venv/bin/activate` to activate the virtual environment you just created. At this point, you will need to install dependencies with `pip`. These include: `torch`, `numpy`, `matplotlib`, and `pandas`. If there are additional dependencies requested by the code when you run it later, download those as well.

## Next Steps
See each folder's `README.md` for additional instructions for running the various models.