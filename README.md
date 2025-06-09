# About
This repository contains code written for the Shirley Ryan AbilityLab Pons Group. 

The goal of the project is to further developments on patient-therapist interactions involving use of lower-body exoskeletons through application of machine-learning models.

This code is authored by Grayson Snyder as part of his final project for the MS in Robotics degree from Northwestern University.

# Quickstart
To utilize this repository, first clone it into a fresh folder. The code requires users to have access to patient and therapist gait data in the form of CSVs with columns labeled `TimeInteractionSubscription, JointPositions_1, JointPositions_2, JointPositions_3, JointPositions_4` for models in the folders `lstm_4to1`, `lstm_4to4`, and `lstm_singleJoint`. Datasets will not be included in this respository as, though de-identified from any patient information, they are best kept private.

See each folder's `README.md` for additional instructions for running the various models.