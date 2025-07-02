import rospy
import torch
import numpy as np
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import time
import argparse
#import torch.nn as nn
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import math
import pickle
import sys
import os

class ROSInterface():
    def __init__(self, args):
        # Initialize ROS publishers and subscribers
        self.diablo_ego_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.diablo_follower_pub = rospy.Publisher('/cmd_vel_follower', Twist, queue_size=10)
        self.pose_sub_1 = rospy.Subscriber('/mocap_node/DiabloEgo/pose', PoseStamped, self.pose_callback_1, queue_size=1)
        self.pose_sub_2 = rospy.Subscriber('/mocap_node/DiabloFollower/pose', PoseStamped, self.pose_callback_2, queue_size=1)
        self.pose_sub_3 = rospy.Subscriber('/mocap_node/Box/pose', PoseStamped, self.pose_callback_3, queue_size=1)
        self.diablo_ego_pose = None
        self.diablo_ego_heading = None
        self.diablo_follower_pose = None
        self.diablo_follower_heading = None
        self.box_pose = None
        self.box_heading = None
        self.terminate = False
        self.ego_pose_x_log = []
        self.ego_pose_y_log = []
        self.box_pose_x_log = []
        self.box_pose_y_log = []
        self.log_dict = {}
        self.log_dict['ego_pose'] = []
        self.log_dict['follower_pose'] = []
        self.log_dict['box_pose'] = []
        self.log_dict['states'] = []
        self.log_dict['actions'] = []
        self.log_dict['errors'] = []
        self.exp_name = args.exp_name
        self.cross_track_error_log = []
        self.heading_angle_error_log = []
        # self.ego_pose_x_log = []
        # self.ego_pose_y_log = []
        # print(self.exp_name)
        




        # self.model = RLAgent(args.model_path)  # Load your model
        self.actions = np.zeros(5)  # Dummy init tensor to avoid error

        self.get_reference_trajectory(args.reference_traj)#[::3, :]

    def pose_callback_1(self, msg):
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = euler_from_quaternion(orientation)
        # heading = np.array([(euler[2] + np.pi) % (2 * np.pi)]) # Convert the angle from optitrack to 0~2pi
        heading = np.array([euler[2]])
        # Orientation and pose tensors
        self.diablo_ego_pose = position  # Shape (1, 3)
        self.diablo_ego_heading = heading  # Shape (1, 1)
        #print('Daiblo Ego:', self.diablo_ego_heading)


    def pose_callback_2(self, msg):
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = euler_from_quaternion(orientation)
        # heading = np.array([(euler[2] + np.pi) % (2 * np.pi)]) # Convert the angle from optitrack to 0~2pi
        heading = np.array([euler[2]])
        # Orientation and pose tensors
        self.diablo_follower_pose = position  # Shape (1, 3)
        self.diablo_follower_heading = heading  # Shape (1, 1)
        #print('Daiblo Follower:', self.diablo_follower_heading)

    def pose_callback_3(self, msg):
        position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = euler_from_quaternion(orientation)
        #heading = np.array([(euler[2] + np.pi) % (2 * np.pi)]) # Convert the angle from optitrack to 0~2pi
        heading = np.array([euler[2]])
        # Orientation and pose tensors
        self.box_pose = position  # Shape (1, 3)
        self.box_heading = heading  # Shape (1, 1)


    def calc_tracking_error(self,box_pose,box_heading):
        distances = np.linalg.norm(self.ref_traj - box_pose[:2], axis=1)
        self.min_index = np.argmin(distances)
        print(self.min_index)
        if (distances.shape[0] - self.min_index) <= 1:
            self.terminate = True

        # Retrieve the minimum distance
        self.cross_track_error = distances[self.min_index]
        # if self.min_index + 1 > 25:
        #     self.min_index = 24 
        desired_heading_angle = np.arctan2((self.ref_traj[self.min_index+1, 1]-self.ref_traj[self.min_index, 1]),(self.ref_traj[self.min_index+1, 0]-self.ref_traj[self.min_index, 0]))
        self.heading_angle_error = desired_heading_angle - box_heading

        temp_ = [self.cross_track_error,self.heading_angle_error]
        self.log_dict['errors'].append(temp_)
        # Logging errors for verification
        # rospy.loginfo(self.cross_track_error)
        # rospy.loginfo(self.heading_angle_error)


    def get_reference_trajectory(self,reference_traj):
        coordinates = np.loadtxt(
            reference_traj,
            delimiter=',', skiprows=1) # Take up to 26  rows[0:26]
        x_ref = coordinates[:,0]
        y_ref = coordinates[:,1]
        self.ref_traj = np.column_stack((x_ref,y_ref))


    def get_follower_commands(self,ego_heading,follower_heading,ego_pose,follower_pose):
        theta_e = ego_heading
        theta_f = follower_heading

        xe,ye = ego_pose[0],ego_pose[1]

        xf,yf = follower_pose[0],follower_pose[1]

        theta_dot = self.actions[4]
        
        vx = self.actions[2]; vy = self.actions[3]         

        A = np.cos(theta_e - theta_f)
        B = theta_dot * (yf * np.cos(theta_e) + xe * np.sin(theta_e)
                         - ye * np.cos(theta_e) - xf * np.sin(theta_e))
        C = np.sin(theta_f - theta_e)
        D = theta_dot * (xe * np.cos(theta_e) - xf * np.cos(theta_e)
                         + ye * np.sin(theta_e) - yf * np.sin(theta_e))
        temp_ = ((vx + vy) - (B + D)) / (A + C)
        # v_follower = np.clip(temp_, min=-1.0, max=1.0)
        v_follower = temp_[0]
        return v_follower

    def publish_zero_velocities(self):
        """Publish zero velocities to stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.z = 0.0
        twist.angular.z = 0.0

        self.diablo_ego_pub.publish(twist)
        self.diablo_follower_pub.publish(twist)
        rospy.loginfo("Published zero velocities on shutdown.")    

    def logger(self):
        # Create a directory
        curr_dir = os.getcwd()
        save_dir = os.path.join(curr_dir,self.exp_name)
        try:
            os.makedirs(save_dir)
            print(f"Folder for run '{self.exp_name}' created")
        except FileExistsError:
            print("Folder already exists")
            exit()
        except Exception as e:
            print("Error encountered creating folder")
            exit()

        
            
        # Save ego pose
        ego_pose_log = np.vstack(self.log_dict['ego_pose'])
        np.savetxt(save_dir+'/ego_pose.csv',ego_pose_log,delimiter=",")
        follower_pose_log = np.vstack(self.log_dict['follower_pose'])
        np.savetxt(save_dir+'/follower_pose.csv',follower_pose_log,delimiter=",")
        box_pose_log = np.vstack(self.log_dict['box_pose'])
        np.savetxt(save_dir+'/follower_pose.csv',box_pose_log,delimiter=",")
        states_log = np.vstack(self.log_dict['states'])
        np.savetxt(save_dir+'/states.csv',states_log,delimiter=",")
        actions_log = np.vstack(self.log_dict['actions'])
        np.savetxt(save_dir+'/actions.csv',actions_log,delimiter=",")
        errors_log = np.vstack(self.log_dict['errors'])
        np.savetxt(save_dir+'/errors.csv',errors_log,delimiter=",")

        # self.log_dict['actions'] = []
        # self.log_dict['errors'] = []
        



    def control_loop(self):
        rate = rospy.Rate(50)  # Control frequency (Hz)
       
        rospy.on_shutdown(self.publish_zero_velocities)
        while not rospy.is_shutdown():
            if self.diablo_ego_pose is not None and self.terminate == False:
                ''' Add all the other states you want'''
                # Taking snapshots of poses which will be sent to all functions and as a state
                ego_pose = self.diablo_ego_pose; ego_heading = self.diablo_ego_heading

                # Log poses
                self.ego_pose_x_log.append(ego_pose[0])
                self.ego_pose_y_log.append(ego_pose[1])

                follower_pose = self.diablo_follower_pose; follower_heading = self.diablo_follower_heading
                box_pose = self.box_pose; box_heading = self.box_heading
                self.box_pose_x_log.append(box_pose[0])
                self.box_pose_y_log.append(box_pose[1])

                self.log_dict['ego_pose'].append(np.hstack((ego_pose,ego_heading)))
                self.log_dict['follower_pose'].append(np.hstack((follower_pose,follower_heading)))
                self.log_dict['box_pose'].append(np.hstack((box_pose,box_heading)))

                self.distance_robots = np.sqrt((follower_pose[0]-ego_pose[0])**2+(follower_pose[1]-ego_pose[1])**2)
                self.diablo1_relative = np.array([box_pose]) - np.array([ego_pose])
                self.diablo2_relative = np.array([box_pose]) - np.array([follower_pose])
                # print('Robot distance:',self.distance_robots)

                # Calculate tracking error
                self.calc_tracking_error(box_pose,box_heading)
                self.heading_angle_error_log.append(self.heading_angle_error)
                self.cross_track_error_log.append(self.cross_track_error)
                x_d1 = ego_pose[0]
                y_d1 = ego_pose[1]
                theta_d1 = ego_heading

                x_o = box_pose[0]
                y_o = box_pose[1]
                theta_o = box_heading

                x_d2 = follower_pose[0]
                y_d2 = follower_pose[1]
                theta_d2 = follower_heading
            
                relative_pos_dox = x_o*math.cos(theta_d1)  - x_d1*math.cos(theta_d1) - y_d1*math.sin(theta_d1) + y_o*math.sin(theta_d1)
                relative_pos_doy = y_o*math.cos(theta_d1)  - y_d1*math.cos(theta_d1) + x_d1*math.sin(theta_d1) - x_o*math.sin(theta_d1)

                relative_pos_d2ox = x_d2*math.cos(theta_o)  - x_o*math.cos(theta_o) - y_o*math.sin(theta_o) + y_d2*math.sin(theta_o)
                relative_pos_d2oy = y_d2*math.cos(theta_o)  - y_o*math.cos(theta_o) + x_o*math.sin(theta_o) - x_d2*math.sin(theta_o)

                relative_heading_d1 = theta_d1 - theta_o
                relative_heading_d2 = theta_o - theta_d2

                relative_heading_d1d2 = theta_d1 - theta_d2
                relative_pos_d1d2x = x_d2*math.cos(theta_d1)  - x_d1*math.cos(theta_d1) - y_d1*math.sin(theta_d1) + y_d2*math.sin(theta_d1)
                relative_pos_d1d2y = y_d2*math.cos(theta_d1)  - y_d1*math.cos(theta_d1) + x_d1*math.sin(theta_d1) - x_d2*math.sin(theta_d1)


                self.state = np.concatenate(
                        [
                            obs_array
                            for obs_array in (
                                np.array([self.distance_robots]),  # Ensure it's 1D
                                self.heading_angle_error,
                                np.array([self.cross_track_error]),  # Ensure it's 1D
                                self.actions,
                                np.array([relative_heading_d1]).reshape(-1),
                                np.array([relative_heading_d2]).reshape(-1),
                                np.array([relative_pos_dox]).reshape(-1),
                                np.array([relative_pos_doy]).reshape(-1),
                                np.array([relative_pos_d2ox]).reshape(-1),
                                np.array([relative_pos_d2oy]).reshape(-1),
                                np.array([relative_heading_d1d2]).reshape(-1),
                                np.array([relative_pos_d1d2x]).reshape(-1),
                                np.array([relative_pos_d1d2y]).reshape(-1),
                            )
                            if obs_array is not None
                        ],
                        axis=-1
                    ).reshape(1, 17)  # Reshape to (1, 17)
                self.log_dict['states'].append(self.state)
                ort_model = ort.InferenceSession("/home/dhruvm-1074/CoordinatedControl_BipedWheeled/policy_090824.onnx")
                # rospy.loginfo(self.cross_track_error)
                # rospy.loginfo(self.heading_angle_error)
                outputs_ONNX = ort_model.run(
                    None,
                    {"obs": self.state.astype(np.float32)},
                )[0]    

                self.actions = outputs_ONNX[0]  # Predict action from the ONNX model
                self.log_dict['actions'].append(self.actions)
                # print(self.actions)
                self.actions[0] = np.clip(self.actions[0], -0.5, 0.5)
                self.actions[1] = np.clip(self.actions[1], -1.0, 1.0)  
                self.actions[2] = np.clip(self.actions[2], -0.5,0.5)
                self.actions[3] = np.clip(self.actions[3], -0.5, 0.5)
                self.actions[4] = np.clip(self.actions[4], -1.0,1.0)
                

                # Change this script for sneding messages to Diablos
                # Create and publish the Twist message for Diablo 1
                twist_msg_1 = Twist()
                twist_msg_1.linear.x = self.actions[0]  # Example: first element is linear velocity in x
                twist_msg_1.angular.z = self.actions[1] # Example: second element is angular velocity around z-axis
                # rospy.loginfo('Ego messages')
                # rospy.loginfo(twist_msg_1.linear.x)
                # rospy.loginfo(twist_msg_1.angular.z)
                

                v_follower = self.get_follower_commands(ego_heading,follower_heading,ego_pose,follower_pose)
                

                twist_msg_2 = Twist()
                twist_msg_2.linear.x = v_follower # Example: first element is linear velocity in x
                twist_msg_2.angular.z = self.actions[4] # Example: second element is angular velocity around z-axis
                # rospy.loginfo('Follower (lead) messages')
                # rospy.loginfo(twist_msg_2.linear.x)
                # rospy.loginfo(twist_msg_1.angular.z)

                # Publish the message
                self.diablo_ego_pub.publish(twist_msg_1)
                self.diablo_follower_pub.publish(twist_msg_2)


            rate.sleep()  # 10 Hz default


        # Clause for shutting down and dumping data in a csv file
        print('Shut down sequence')
        twist_msg_1.linear.x = 0
        twist_msg_1.angular.z = 0
        twist_msg_2.linear.x = 0
        twist_msg_2.angular.z = 0
        self.diablo_ego_pub.publish(twist_msg_1)
        self.diablo_follower_pub.publish(twist_msg_2)
        print('zero vel published')

        # Logger call
        if self.exp_name != 'None':
            self.logger()
        else:
            print('This run was not saved')

        plt.plot(self.ref_traj[:self.min_index,0],self.ref_traj[:self.min_index,1])
        plt.plot(self.box_pose_x_log[:],self.box_pose_y_log[:],'--b')
        plt.plot(self.ego_pose_x_log[:],self.ego_pose_y_log[:],'--g')
        plt.legend(['Reference','Box pose','Ego pose'])
        # plt.plot(self.cross_track_error_log, '--b')
        # plt.plot(self.heading_angle_error_log)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Provide input parameters for script')
    parser.add_argument('--model_path', type=str, help='Path for pre-trained pytorch model')
    parser.add_argument('--reference_traj', type=str, help='Path for reference trajectory')
    parser.add_argument('--exp_name', type=str, help='Prefix for all log files')
    args = parser.parse_args()
    rospy.init_node('RLController')
    ros_interface = ROSInterface(args)
    ros_interface.control_loop()


if __name__ == "__main__":
    main()

