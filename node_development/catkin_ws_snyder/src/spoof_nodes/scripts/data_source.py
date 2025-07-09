import rospy
import pandas as pd
from std_msgs.msg import String
from spoof_nodes.msg import patient_data
import numpy as np

# First Subject-Therapist Pair
patient1_datapath = '/home/cerebro/snyder_project/data/Patient1_X2_SRA_A_07-05-2024_10-39-10.csv'
patient1_part1 = pd.read_csv(patient1_datapath, 
                          skiprows=range(1, 229607), 
                          nrows=433021-229607)
patient1_part2 = pd.read_csv(patient1_datapath, 
                          skiprows=range(1, 516255), 
                          nrows=718477-516255)
patient1_part3 = pd.read_csv(patient1_datapath, 
                          skiprows=range(1, 761002), 
                          nrows=960356-761002)

# Second Subject-Therapist Pair
patient2_datapath = '/home/cerebro/snyder_project/data/Patient2_X2_SRA_A_08-05-2024_14-33-44.csv'
patient2_part1 = pd.read_csv(patient2_datapath,
                            skiprows=range(1, 123920),
                            nrows=272301-123920)
patient2_part2 = pd.read_csv(patient2_datapath, 
                     skiprows=range(1, 457108), 
                     nrows=595193-457108)

# Third Subject-Therapist Pair
patient3_datapath = '/home/cerebro/snyder_project/data/Patient3_X2_SRA_A_29-05-2024_13-36-40.csv'
patient3_part1 = pd.read_csv(patient3_datapath, 
                          skiprows=range(1, 7694), 
                          nrows=198762-7694)
patient3_part2 = pd.read_csv(patient3_datapath, 
                          skiprows=range(1, 232681), 
                          nrows=388280-232681)
patient3_part3 = pd.read_csv(patient3_datapath, 
                          skiprows=range(1, 417061), 
                          nrows=606461-417061)


p1 = pd.concat([patient1_part1, patient1_part2, patient1_part3])
p2 = pd.concat([patient2_part1, patient2_part2])
p3 = pd.concat([patient3_part1, patient3_part2, patient3_part3])

train_1_len = len(p1) * 0.7
validate_1_len = len(p1) * 0.2
train_2_len = len(p2) * 0.7
validate_2_len = len(p2) * 0.2
train_3_len = len(p3) * 0.7
validate_3_len = len(p3) * 0.2

# Split testing portion of data from full dataset
patient1_test = p1[int(train_1_len + validate_1_len):]

patient2_test = p2[int(train_2_len + validate_2_len):]

patient3_test = p3[int(train_3_len + validate_3_len):]

patient1_data = patient1_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                              ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4',
                              ' StanceInterpolationFactor', ' BackPackAngle', ' BackPackAngularVelocity']].values.astype('float32')

patient2_data = patient2_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                               ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4',
                               ' StanceInterpolationFactor', ' BackPackAngle', ' BackPackAngularVelocity']].values.astype('float32')

patient3_data = patient3_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4',
                               ' JointVelocities_1', ' JointVelocities_2', ' JointVelocities_3', ' JointVelocities_4',
                               ' StanceInterpolationFactor', ' BackPackAngle', ' BackPackAngularVelocity']].values.astype('float32')


def patient_raw():
    pub = rospy.Publisher('patient_data', patient_data, queue_size=10)
    rospy.init_node('patient_raw', anonymous=True)
    rate = rospy.Rate(333)
    i = 0
    while not rospy.is_shutdown():
        if i >= len(patient1_data):
            i = 0
        msg = prep_message(i)
        pub.publish(msg)
        rate.sleep()


def prep_message(index):
    msg = patient_data()
    msg.JointPositions_1 = patient1_data[index][0]
    msg.JointPositions_2 = patient1_data[index][1]
    msg.JointPositions_3 = patient1_data[index][2]
    msg.JointPositions_4 = patient1_data[index][3]
    msg.JointVelocities_1 = patient1_data[index][4]
    msg.JointVelocities_2 = patient1_data[index][5]
    msg.JointVelocities_3 = patient1_data[index][6]
    msg.JointVelocities_4 = patient1_data[index][7]
    msg.StanceInterpolationFactor = patient1_data[index][8]
    msg.BackPackAngle = patient1_data[index][9]
    msg.BackPackAngularVelocity = patient1_data[index][10]
    return msg

if __name__ == '__main__':
    try:
        patient_raw()
    except rospy.ROSInterruptException:
        pass