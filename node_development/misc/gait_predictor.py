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
import yaml


# Define Node
class ModelNode:
    def __init__(self):
        rospy.init_node('model_node', anonymous=True)

        # Initialize RingBuffer
        self.buffer = RingBuffer(50)

        # Check GPU availability
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")

        # Load model configurations
        config_path = os.path.join(os.path.dirname(__file__), 'lstm_models/model_configs.yaml')
        with open(config_path) as f:
            self.model_configs = yaml.safe_load(f)['models']

        # Load default model config
        self.current_model = 'trial5'
        self.model = None
        self.input_slice = None
        self.load_model(self.current_model)

        # Initialize parameters
        self.init_time = time.time()
        self.previous_pred = [None, None, None, None]
        self.config = {
            'pred_diff_threshold': 1.0,
            'model_path': '/home/cerebro/catkin_ws/src/CANOpenRobotController/python/lstm_models/lstm_trial5.pt'
        }
        self.model_path = '/home/cerebro/catkin_ws/src/CANOpenRobotController/python/lstm_models/lstm_trial5.pt'
        self.skipped_pred_count = 0

        # Initialize dynamic reconfigure
        self.server = Server(lstm_model_paramsConfig, self.reconfigure_callback)

        # Load model
        self.model = torch.jit.load(self.model_path).to(self.device)
        self.model.eval()

        # Prepare Subscriber, Synchronizer, and Publisher
        robot_name = rospy.get_param('~robot_name', 'X2_SRA_A')  # Default to 'X2_SRA_A' if not set
        js_sub = Subscriber('/'+robot_name+'/joint_states', JointState)
        si_sub = Subscriber('/'+robot_name+'/stance_interpol', X2StanceInterpolation)
        synchronizer = ApproximateTimeSynchronizer([js_sub, si_sub], queue_size=10, slop=0.0001)
        synchronizer.registerCallback(self.callback)
        self.pub = rospy.Publisher('/'+str(robot_name)+'/walking_mode_SS_kinematic/', Float64MultiArray, queue_size=10)

    def run(self):
        rospy.spin()

    def load_model(self, model_name):
        if model_name not in self.model_configs:
            rospy.logerr(f"Model {model_name} not found in configurations.")
            return False
        config = self.model_configs[model_name]
        try:
            model_path = os.path.join(os.path.dirname(__file__), config['path'])
            self.model = torch.jit.load(model_path).to(self.device)
            self.model.eval()

            self.input_slice = config['input_slice']
            self.current_model = model_name
            # self.buffer.clear()
            return True
        except Exception as e:
            rospy.logerr(f"Failed to load model {model_name} from {model_path}: {e}")
            self.model = None
            self.input_slice = None
            return False

    def predict(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            y_pred = self.model(x)
        return y_pred

    def reconfigure_callback(self, config, level):
        rospy.loginfo("""Reconfigure Request: {pred_diff_threshold}, {model_type}""".format(**config))
        model_mapping = {
            0: "trial4",
            1: "trial5",
            2: "trial6"
        }
        try: 
            if config['model_type'] in model_mapping:
                model_name = model_mapping[config['model_type']]
                if model_name != self.current_model:
                    if self.load_model(model_name):
                        rospy.loginfo(f"Changing model to {model_name}")
                    else:
                        config['model_type'] = self.current_model  # Revert to previous model if loading fails
                        rospy.logerr(f"Failed to load model {model_name}, reverting to {self.current_model}")
                self.config['pred_diff_threshold'] = config['pred_diff_threshold']
                self.input_slice = self.model_configs[model_name]['input_slice']
                # self.buffer.clear()
            return config
        except Exception as e:
            rospy.logerr(f"Error in reconfigure callback: {e}")
            return config

    def callback(self, js_data, si_data):
        try:
            # Extract and combine data
            data = torch.tensor([js_data.position[0], js_data.position[1], js_data.position[2], js_data.position[3],
                                js_data.velocity[0], js_data.velocity[1], js_data.velocity[2], js_data.velocity[3],
                                si_data.treadmill, js_data.position[4], js_data.velocity[4]], dtype=torch.float32)
            
            # # Verify incoming data is changing
            # if len(self.buffer.data) > 0:
            #     data_diff = torch.sum(torch.abs(data - self.buffer.data[-1]))
            #     rospy.loginfo(f"Data difference from last sample: {data_diff.item()}")

            # Append to buffer
            self.buffer.append(data)

            # Check if we have enough data to make a prediction
            if len(self.buffer.data) < self.buffer.max:
                return # Not enough data to make a prediction
           
            # Prepare input data
            # Shape data for prediction based on input_slice for model
            buffer_data = torch.stack(list(self.buffer.get()))
            shaped_data = buffer_data[:, self.input_slice]
            input_data = shaped_data.unsqueeze(0) # Add batch dimension
            
            # # Verify temporal variation
            # rospy.loginfo(f"Temporal variation across buffer:\n"
            #             f"First: {shaped_data[0]}\n"
            #             f"Middle: {shaped_data[len(shaped_data)//2]}\n"
            #             f"Last: {shaped_data[-1]}")

            # Predict
            start_time = time.time()
            y_pred = self.predict(input_data)[:, -1, :].cpu().squeeze()
            end_time = time.time()
            
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
                    writer.writerow([self.current_model, end_time-start_time, y_pred[0].item(), y_pred[1].item(), y_pred[2].item(), y_pred[3].item(),
                                    js_data.position[0], js_data.position[1], js_data.position[2], js_data.position[3],
                                    js_data.velocity[0], js_data.velocity[1], js_data.velocity[2], js_data.velocity[3],
                                    si_data.treadmill, js_data.position[4], js_data.velocity[4]])
        except Exception as e:
            rospy.logerr(f"Error in synch callback: {e}")


if __name__ == '__main__':
    try:
        model_node = ModelNode()
        model_node.run()
    except rospy.ROSInterruptException:
        pass