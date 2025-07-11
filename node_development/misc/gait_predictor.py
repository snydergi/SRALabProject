#!/usr/bin/env python3
import rospy
import os
import csv
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from CORC.msg import X2StanceInterpolation
import time
from RingBuffer import RingBuffer
import torch
import torch.nn as nn
import dynamic_reconfigure.client


# Model definition
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 4)
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)


# Define Node
class ModelNode:
    def __init__(self):
        rospy.init_node('model_node', anonymous=True)
        client = dynamic_reconfigure.client.Client('model_node')
        params = {'model_path': ['/lstm_models/trial5_model_epoch198.pth']}
        config = client.update_configuration(params)
        self.init_time = time.time()

        # Initialize RingBuffer
        self.buffer = RingBuffer(50)

        # Load model
        self.model = JointModel()
        self.model.load_state_dict(torch.load('/home/cerebro/snyder_project/SRALabProject/lstm_BigData/trial5/lstm_model_epoch198.pth'))
        self.model.eval()

        # Prepare Subscriber, Synchronizer, and Publisher
        robot_name = rospy.get_param('~robot_name', 'X2_SRA_A')  # Default to 'X2_SRA_A' if not set
        js_sub = Subscriber('/'+robot_name+'/joint_states', JointState)
        si_sub = Subscriber('/'+robot_name+'/stance_interpol', X2StanceInterpolation)
        synchronizer = ApproximateTimeSynchronizer([js_sub, si_sub], queue_size=10, slop=0.001)
        synchronizer.registerCallback(self.callback)
        self.pub = rospy.Publisher('/'+str(robot_name)+'/walking_mode_SS_kinematic/', Float64MultiArray, queue_size=10)

    def run(self):
        rospy.spin()
    def predict(self, x):
        y_pred = self.model(x)
        return y_pred
    def callback(self, js_data, si_data):
        try:
            # Extract and combine data
            data = torch.tensor([js_data.position[0], js_data.position[1], js_data.position[2], js_data.position[3],
                                js_data.velocity[0], js_data.velocity[1], js_data.velocity[2], js_data.velocity[3],
                                si_data.treadmill, js_data.position[4], js_data.velocity[4]], dtype=torch.float32)
            self.buffer.append(data)
            if len(self.buffer.get()) < self.buffer.max:
                pass # Not enough data to make a prediction
            else:
                # Prepare input data
                input_data = torch.stack(self.buffer.data)
                input_data = input_data.unsqueeze(0) # Add batch dimension
                # Predict
                y_pred = self.predict(input_data)[:, -1, :].squeeze()
                # Prepare message
                msg = Float64MultiArray()
                msg.data.append(y_pred[0].item())
                msg.data.append(y_pred[1].item())
                msg.data.append(y_pred[2].item())
                msg.data.append(y_pred[3].item())
                # Publish message
                self.pub.publish(msg)
                # Log prediction
                with open(f'/home/cerebro/snyder_project/SRALabProject/node_development/misc/pred_data/prediction_{self.init_time}.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([y_pred[0].item(), y_pred[1].item(), y_pred[2].item(), y_pred[3].item(),
                                     js_data.position[0], js_data.position[1], js_data.position[2], js_data.position[3]])
        except Exception as e:
            rospy.logerr(f"Error in synch callback: {e}")


if __name__ == '__main__':
    try:
        model_node = ModelNode()
        model_node.run()
    except rospy.ROSInterruptException:
        pass