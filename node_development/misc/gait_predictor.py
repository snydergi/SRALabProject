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
from dynamic_reconfigure.server import Server
from CORC.cfg import lstm_model_paramsConfig


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
        self.server = Server(lstm_model_paramsConfig, self.reconfigure_callback)

        # Initialize parameters
        self.init_time = time.time()
        self.previous_pred = [None, None, None, None]
        self.config = {
            'pred_diff_threshold': 0.05,
            'model_path': '/home/cerebro/snyder_project/SRALabProject/lstm_BigData/trial5/lstm_model_epoch198.pth'
        }
        self.model_path = '/home/cerebro/snyder_project/SRALabProject/lstm_BigData/trial5/lstm_model_epoch198.pth'
        self.skipped_pred_count = 0

        # Initialize RingBuffer
        self.buffer = RingBuffer(50)

        # Load model
        self.model = JointModel()
        self.model.load_state_dict(torch.load(self.model_path))
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

    def reconfigure_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {pred_diff_threshold}, {model_path}""".format(**config))

        # Reload model if path has changed
        if hasattr(self, 'config') and self.config is not None:
            if config['model_path'] != self.config['model_path']:
                rospy.loginfo("Model path changed to: {}".format(config['model_path']))
                self.model_path = config['model_path']
                # Load new model
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()

        self.config = config

        return config

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
                # Check for too large of a jump in prediction
                if self.previous_pred[0] is not None:
                    if (abs(y_pred[0].item() - self.previous_pred[0]) > self.config['pred_diff_threshold'] or
                        abs(y_pred[1].item() - self.previous_pred[1]) > self.config['pred_diff_threshold'] or
                        abs(y_pred[2].item() - self.previous_pred[2]) > self.config['pred_diff_threshold'] or
                        abs(y_pred[3].item() - self.previous_pred[3]) > self.config['pred_diff_threshold']):
                        rospy.logwarn("Large jump in prediction detected, skipping this prediction.")
                        self.skipped_pred_count += 1
                        if self.skipped_pred_count > 10:
                            self.previous_pred = [None, None, None, None]  # Reset previous prediction
                            self.skipped_pred_count = 0
                        return
                # Update previous prediction
                self.previous_pred = [y_pred[0].item(), y_pred[1].item(), y_pred[2].item(), y_pred[3].item()]
                # Publish message
                self.pub.publish(msg)
                # Log prediction
                with open(f'/home/cerebro/snyder_project/SRALabProject/node_development/misc/pred_data/prediction_{self.init_time}.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([y_pred[0].item(), y_pred[1].item(), y_pred[2].item(), y_pred[3].item(),
                                     js_data.position[0], js_data.position[1], js_data.position[2], js_data.position[3]])
        except Exception as e:
            rospy.logerr(f"Error in synch callback: {e.with_traceback()}")


if __name__ == '__main__':
    try:
        model_node = ModelNode()
        model_node.run()
    except rospy.ROSInterruptException:
        pass