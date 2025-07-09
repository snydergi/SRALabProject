import rospy
from std_msgs.msg import String
from spoof_nodes.msg import patient_data, therapist_pred
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import time

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
        self.pub = rospy.Publisher("therapist_pred", therapist_pred, queue_size=10)
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
            start = time.time()
            y_pred = predict(input_data)[:, -1, :].squeeze()
            end = time.time()
            rospy.loginfo(f"Prediction time: {end - start:.4f} seconds")
            # Prepare message
            msg = therapist_pred()
            msg.JointPositions_1 = y_pred[0].item()
            msg.JointPositions_2 = y_pred[1].item()
            msg.JointPositions_3 = y_pred[2].item()
            msg.JointPositions_4 = y_pred[3].item()
            msg.Therapist_TrueJointPosition_1 = data.Therapist_TrueJointPosition_1
            msg.Therapist_TrueJointPosition_2 = data.Therapist_TrueJointPosition_2
            msg.Therapist_TrueJointPosition_3 = data.Therapist_TrueJointPosition_3
            msg.Therapist_TrueJointPosition_4 = data.Therapist_TrueJointPosition_4
            # Publish message
            self.pub.publish(msg)
            # rospy.loginfo(f"Predicted joint positions: {y_pred}")

if __name__ == '__main__':
    try:
        model_node = ModelNode()
        model_node.run()
    except rospy.ROSInterruptException:
        pass
