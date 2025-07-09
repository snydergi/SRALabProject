import rospy
from std_msgs.msg import String
from spoof_nodes.msg import patient_data, therapist_pred
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# First Therapist
therapist1_datapath = '/home/cerebro/snyder_project/data/Therapist1_X2_SRA_B_07-05-2024_10-41-46.csv'
therapist1_part1 = pd.read_csv(therapist1_datapath, 
                            skiprows=range(1, 165636), 
                            nrows=369050-165636)
therapist1_part2 = pd.read_csv(therapist1_datapath, 
                            skiprows=range(1, 452284), 
                            nrows=654506-452284)
therapist1_part3 = pd.read_csv(therapist1_datapath, 
                            skiprows=range(1, 697032), 
                            nrows=896386-697032)

# Second Therapist
therapist2_datapath = '/home/cerebro/snyder_project/data/Therapist2_X2_SRA_B_08-05-2024_14-33-51.csv'
therapist2_part1 = pd.read_csv(therapist2_datapath,
                            skiprows=range(1, 123920),
                            nrows=272301-123920)
therapist2_part2 = pd.read_csv(therapist2_datapath, 
                       skiprows=range(1, 457108), 
                       nrows=595193-457108)

# Third Therapist
therapist3_datapath = '/home/cerebro/snyder_project/data/Therapist3_X2_SRA_B_29-05-2024_13-41-19.csv'
therapist3_part1 = pd.read_csv(therapist3_datapath, 
                            skiprows=range(1, 7694), 
                            nrows=198762-7694)
therapist3_part2 = pd.read_csv(therapist3_datapath, 
                            skiprows=range(1, 232681), 
                            nrows=388280-232681)
therapist3_part3 = pd.read_csv(therapist3_datapath, 
                            skiprows=range(1, 417062), 
                            nrows=606462-417062)

t1 = pd.concat([therapist1_part1, therapist1_part2, therapist1_part3])
t2 = pd.concat([therapist2_part1, therapist2_part2])
t3 = pd.concat([therapist3_part1, therapist3_part2, therapist3_part3])

train_1_len = len(t1) * 0.7
validate_1_len = len(t1) * 0.2
train_2_len = len(t2) * 0.7
validate_2_len = len(t2) * 0.2
train_3_len = len(t3) * 0.7
validate_3_len = len(t3) * 0.2

# Split testing portion of data from full dataset
therapist1_test = t1[int(train_1_len + validate_1_len):]
therapist2_test = t2[int(train_2_len + validate_2_len):]
therapist3_test = t3[int(train_3_len + validate_3_len):]

therapist1_data = therapist1_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')
therapist2_data = therapist2_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')
therapist3_data = therapist3_test[[' JointPositions_1', ' JointPositions_2', ' JointPositions_3', ' JointPositions_4']].values.astype('float32')

j1 = []
j2 = []
j3 = []
j4 = []

# Model definition
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 4)
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)

# Load model
model = JointModel()
model.load_state_dict(torch.load('/home/cerebro/snyder_project/SRALabProject/lstm_BigData/trial5/lstm_model_epoch198.pth'))
model.eval()

# From CANOpenRobotController
class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full


buffer = RingBuffer(50)

def predict(x):
    y_pred = model(x)
    return y_pred

class ModelNode:
    def __init__(self):
        rospy.init_node('model_node', anonymous=True)
        self.plotted = False
        self.patient_number = 1
        rospy.Subscriber("patient_data", patient_data, self.callback)
    def run(self):
        rospy.spin()
    def callback(self, data):
        if len(buffer.data) < buffer.max:
            buffer.append(data)
        else:
            buffer.append(data)
            # Prepare input data
            input_data = torch.tensor([[data.JointPositions_1, data.JointPositions_2, data.JointPositions_3, data.JointPositions_4,
                                        data.JointVelocities_1, data.JointVelocities_2, data.JointVelocities_3, data.JointVelocities_4,
                                        data.StanceInterpolationFactor, data.BackPackAngle, data.BackPackAngularVelocity]], dtype=torch.float32)
            input_data = input_data.unsqueeze(0)  # Add batch dimension
            # Predict
            y_pred = predict(input_data)[:, -1, :].squeeze()
            j1.append(y_pred[0].item())
            j2.append(y_pred[1].item())
            j3.append(y_pred[2].item())
            j4.append(y_pred[3].item())
            # rospy.loginfo(f"Predicted joint positions: {y_pred}")
            if len(j1) > 5000 and not self.plotted:
                self.plotted = True
                fig = plt.figure(figsize=(12, 6))
                fig.suptitle(f"Patient {self.patient_number} Testing: Therapist Predictions from Patient Data")
                plt.subplot(2, 2, 1)
                plt.plot(therapist1_data[:, 0], c='b', label=f'True Therapist Data')
                plt.plot(j1, c='r', linestyle='--', label=f'Predicted Therapist Data')
                plt.xlim(0, 7500)
                plt.xlabel('Time Steps (~4ms)')
                plt.ylabel('Joint Positions (Radians)')
                plt.legend()
                plt.title("Joint 1")

                plt.subplot(2, 2, 2)
                plt.plot(therapist1_data[:, 1], c='b', label=f'True Therapist Data')
                plt.plot(j2, c='r', linestyle='--', label=f'Predicted Therapist Data')
                plt.xlim(0, 7500)
                plt.xlabel('Time Steps (~4ms)')
                plt.ylabel('Joint Positions (Radians)')
                plt.legend()
                plt.title("Joint 2")

                plt.subplot(2, 2, 3)
                plt.plot(therapist1_data[:, 2], c='b', label=f'True Therapist Data')
                plt.plot(j3, c='r', linestyle='--', label=f'Predicted Therapist Data')
                plt.xlim(0, 7500)
                plt.xlabel('Time Steps (~4ms)')
                plt.ylabel('Joint Positions (Radians)')
                plt.legend()
                plt.title("Joint 3")

                plt.subplot(2, 2, 4)
                plt.plot(therapist1_data[:, 3], c='b', label=f'True Therapist Data')
                plt.plot(j4, c='r', linestyle='--', label=f'Predicted Therapist Data')
                plt.xlim(0, 7500)
                plt.xlabel('Time Steps (~4ms)')
                plt.ylabel('Joint Positions (Radians)')
                plt.legend()
                plt.title("Joint 4")
                
                plt.show()

if __name__ == '__main__':
    try:
        model_node = ModelNode()
        model_node.run()
    except rospy.ROSInterruptException:
        pass
