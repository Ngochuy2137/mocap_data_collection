#!/usr/bin/env python3

'''
--------------------------------------------------------------------------------------------------------
Program description
    - Subscribes to a PoseStamped topic and records the trajectory of the object.
    - Start recording when the user presses ENTER 
    -   Interpolate velocity based on central difference method, not backward/forward difference method.
    -       I interpolate velocities manually instead of using np.gradient() because np.gradient() auto interpolates if time steps are not equal.
    - Swap y and z if necessary.
    - Stop when the object reaches a certain height.
    - Log a warning if the message frequency is lower than a certain threshold.
    - Save the recorded trajectories to a file named 'trajectories.npz' when the user stops the program.
    -     Data format:  [  
                            [   
                                points: [
                                            [x, y, z, vx, vy, vz, timestamp], 
                                            ...
                                        ], 
                                low_freq_num: int
                            ], 
                            ...
                        ]

Maintainer: Huynn
--------------------------------------------------------------------------------------------------------
'''


import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
import time
import threading
import os

# np.set_printoptions(precision=3)
# Thresholds
MIN_HEIGHT_THRESHOLD = 0.1  # Trajectory end height threshold
MIN_FREQ_THRESHOLD = 100  # Minimum message frequency (Hz)
MOCAP_OBJECT_TOPIC = '/mocap_pose_topic/frisbee1_pose'
SWAP_YZ = False
DECIMAL_NUM = 5
# Global vars
trajectories = []
current_trajectory = []
low_freq_count = 0
recording = False
start_time = time.strftime("%d-%m-%Y_%H-%M-%S")     # get time now: d/m/y/h/m
recording_lock = threading.Lock()

def pose_callback(msg:PoseStamped):
    global recording, current_trajectory, low_freq_count
    with recording_lock:
        if recording:
            # Get the current time and calculate the message frequency
            current_timestamp = msg.header.stamp.to_sec()
            if len(current_trajectory) > 0:
                last_timestamp = current_trajectory[-1][3]
                time_diff = (current_timestamp - last_timestamp)
                freq = 1.0 / time_diff
                if freq < MIN_FREQ_THRESHOLD:
                    freq = round(freq, 2)
                    rospy.logwarn("Current message frequency is too low: " + str(freq) + " Hz")
                    low_freq_count += 1
            
            # get x, y, z
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            # Swap y and z
            if SWAP_YZ:
                y, z = z, y
            
            # Save the current position to the current trajectory
            new_point = np.array([x, y, z, current_timestamp])
            current_trajectory.append(new_point)

            # print out
            new_point_prt = [round(i, DECIMAL_NUM) for i in new_point]
            print('     New point: x=', new_point_prt[0], ' y=', new_point_prt[1], ' z=', new_point_prt[2], ' t=', new_point_prt[3], ' low_freq_count=', low_freq_count)

            # Check end condition of the current trajectory
            if z < MIN_HEIGHT_THRESHOLD:
                rospy.loginfo("Trajectory ended with " + str(len(current_trajectory)) + " points !")
                process_trajectory()
                recording = False

def vel_interpolation(x_arr, t_arr):
    vel = []
    for i in range(len(x_arr)):
        if i == 0:
            # Forward difference
            vel_i = (x_arr[i + 1] - x_arr[i]) / (t_arr[i + 1] - t_arr[i])
        elif i == len(x_arr) - 1:
            # Backward difference
            vel_i = (x_arr[i] - x_arr[i - 1]) / (t_arr[i] - t_arr[i - 1])
        else:
            # Central difference
            vel_i = (x_arr[i + 1] - x_arr[i - 1]) / (t_arr[i + 1] - t_arr[i - 1])
        vel.append(vel_i)
    vel = np.array(vel)
    return vel

def process_trajectory():
    global trajectories, current_trajectory, low_freq_count
    # Interpolate vx, vy, vz from x, y, z
    data_points_np = np.array(current_trajectory)
    if len(data_points_np) > 1:
        vx = vel_interpolation(data_points_np[:, 0], data_points_np[:, 3])  # vx = dx/dt
        vy = vel_interpolation(data_points_np[:, 1], data_points_np[:, 3])  # vy = dy/dt
        vz = vel_interpolation(data_points_np[:, 2], data_points_np[:, 3])  # vz = dz/dt
        zeros = np.zeros_like(vx)
        gravity = np.full_like(vx, 9.81)

        print('\n-----------------')
        print('data_points_np[:, :3] shape: ', data_points_np[:, :3].shape)
        print('vx shape: ', vx.shape)
        print('vy shape: ', vy.shape)
        print('vz shape: ', vz.shape)
        print('zeros shape: ', zeros.shape)
        print('gravity shape: ', gravity.shape)
        extened_data_points = np.column_stack((data_points_np[:, :3], vx, vy, vz, zeros, zeros, gravity))
        # extened_data_points = np.round(extened_data_points, DECIMAL_NUM)
        trajectory_data = {
            'points': extened_data_points,
            'time_stamps': data_points_np[:, 3],
            'low_freq_num': low_freq_count
        }
        trajectories.append(trajectory_data)

        # Save all trajectories to file after each completed trajectory
        save_trajectories_to_file()
    else:
        rospy.logwarn("The trajectory needs to have at least 2 points to be saved !")
    
    # Reset current trajectory
    current_trajectory = []
    low_freq_count = 0

def save_trajectories_to_file():
    global trajectories, start_time

    # Get the path of the current directory (where the script is running)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Move one directory up
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # Create a path to the directory ../trajectories
    trajectories_dir = os.path.join(parent_dir, 'data')
    # Create directory ../trajectories if it does not exist
    if not os.path.exists(trajectories_dir):
        os.makedirs(trajectories_dir)

    file_name = 'trajectories_' + str(start_time) + '.npz'
    file_path = os.path.join(trajectories_dir, file_name)
    
    # Save trajectories using numpy npz format
    trajectory_dict = {'trajectories': trajectories}
    np.savez(file_path, **trajectory_dict)  # Save each trajectory as a key-value array
    rospy.loginfo("All atrajectories has been saved to file " + file_name)

def check_user_input():
    global recording
    while not rospy.is_shutdown():
        with recording_lock:
            if not recording:
                print('\n\n------------------------- [', len(trajectories), '] -------------------------')
                input("Press ENTER to start new trajectory collection ...")
                recording = True
                rospy.loginfo("Collecting ...\n")

if __name__ == '__main__':
    rospy.init_node('trajectory_recorder', anonymous=True)

    # Subscribe vào topic pose
    rospy.Subscriber(MOCAP_OBJECT_TOPIC, PoseStamped, pose_callback)

    # Thread để chờ input của người dùng
    user_input_thread = threading.Thread(target=check_user_input)
    user_input_thread.daemon = True
    user_input_thread.start()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    # finally:
    #     # Lưu tất cả quỹ đạo vào file khi dừng chương trình
    #     save_trajectories_to_file()
