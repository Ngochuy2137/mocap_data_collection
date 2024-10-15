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
                                            [x, y, z, vx, vy, vz, 0, 0, 9.81], 
                                            ...
                                        ], 
                                time_stamps: [double],
                                low_freq_num: int
                            ], 
                            ...
                        ]
    - Some other features:
        - Avoiding messages missing:
            - Proper ENTER checking:
                - To get ENTER input from user, I create a thread for this task. However, when ENTER checking thread run continuously, along with main thread, 
                it might lock the main thread and cause missing messages (check self.recording_lock variable)
                - Therefore, to avoid missing message, I use threading.Event to activate/deactivate thread that check user ENTER input.
                - The ENTER checking thread is activated only after finishing a new trajectory or when starting the program.
            - Message queueing problem avoidance:
                - When waiting for user to press ENTER, the program should not queue messages. Therefore, check the lock before processing messages (check self.recording_lock.acquire(blocking=False)):
                    - If self.recording_lock.acquire(blocking=False), no locking, process the message.
                    - If not, skip the message.

Maintainer: Huynn
--------------------------------------------------------------------------------------------------------
'''

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
import time
import threading
import os

class RoCatDataCollector:

    def __init__(self, mocap_object_topic, min_height_threshold, min_freq_threshold, swap_yz):
        self.decima_prt_num = 5
        self.start_time = time.strftime("%d-%m-%Y_%H-%M-%S")     # get time now: d/m/y/h/m
        self.number_subscriber = rospy.Subscriber(mocap_object_topic, PoseStamped, self.pose_callback)
        self.mocap_object_topic = mocap_object_topic

        self.min_height_threshold = min_height_threshold
        self.min_freq_threshold = min_freq_threshold
        self.swap_yz = swap_yz
        self.collected_data = []
        self.reset()

        # Start user input thread
        self.enter_event = threading.Event()
        self.enter_event.set()  # Kích hoạt thread để kiểm tra ENTER
        self.recording_lock = threading.Lock()
        self.user_input_thread = threading.Thread(target=self.check_user_input)
        self.user_input_thread.daemon = True
        self.user_input_thread.start()

    def pose_callback(self, msg:PoseStamped):
        # get x, y, z
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        # Swap y and z
        if self.swap_yz:
            y, z = z, y
        self.current_position = [x, y, z]
        current_timestamp = msg.header.stamp.to_sec()

        # If the lock cannot be obtained, skip the callback
        if not self.recording_lock.acquire(blocking=False):
            # rospy.loginfo("Skipping message due to lock being held.")
            return

        '''
        try...finally ensures that the lock will always be released, no matter what happens during processing inside the try block. 
        Without finally, when an error (or exception) occurs, the lock can be held forever, and no other thread can obtain the lock.
        '''
        try:
            if self.recording:
                # Get the current time and calculate the message frequency
                if len(self.current_trajectory) > 0:
                    last_timestamp = self.current_trajectory[-1][3]
                    time_diff = (current_timestamp - last_timestamp)
                    freq = 1.0 / time_diff
                    if freq < self.min_freq_threshold:
                        freq = round(freq, 2)
                        rospy.logwarn("Current message frequency is too low: " + str(freq) + " Hz")
                        self.low_freq_count += 1
                
                # Save the current position to the current trajectory
                new_point = self.current_position + [current_timestamp]
                self.current_trajectory.append(new_point)

                # print out
                new_point_prt = [round(i, self.decima_prt_num) for i in new_point]
                print('     New point: x=', new_point_prt[0], ' y=', new_point_prt[1], ' z=', new_point_prt[2], ' t=', new_point_prt[3], ' low_freq_count=', self.low_freq_count)

                # Check end condition of the current trajectory
                if self.current_position[2] < self.min_height_threshold:
                    rospy.loginfo("Trajectory ended with " + str(len(self.current_trajectory)) + " points !")
                    if self.process_trajectory():
                        # Save all trajectories to file after each new proper trajectory
                        self.save_trajectories_to_file()
                    self.enter_event.set()  # Kích hoạt thread để kiểm tra ENTER
                    self.reset()
        finally:
            # Giải phóng lock sau khi xử lý xong
            self.recording_lock.release()
                    
    def reset(self,):
        self.current_position = [0, 0, 0]
        self.current_trajectory = []
        self.low_freq_count = 0
        self.recording = False
        

    # Process the current trajectory
    def process_trajectory(self):
        # Interpolate vx, vy, vz from x, y, z
        data_points_np = np.array(self.current_trajectory)
        if len(data_points_np) > 1:
            vx = self.vel_interpolation(data_points_np[:, 0], data_points_np[:, 3])  # vx = dx/dt
            vy = self.vel_interpolation(data_points_np[:, 1], data_points_np[:, 3])  # vy = dy/dt
            vz = self.vel_interpolation(data_points_np[:, 2], data_points_np[:, 3])  # vz = dz/dt
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
                'low_freq_num': self.low_freq_count
            }
            self.collected_data.append(trajectory_data)
            return True
        else:
            rospy.logwarn("The trajectory needs to have at least 2 points to be saved !")
            return False
    
    def vel_interpolation(self, x_arr, t_arr):
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

    def save_trajectories_to_file(self,):
        # Get the path of the current directory (where the script is running)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # Move one directory up
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        # Create a path to the directory ../trajectories
        trajectories_dir = os.path.join(parent_dir, 'data')
        # Create directory ../trajectories if it does not exist
        if not os.path.exists(trajectories_dir):
            os.makedirs(trajectories_dir)

        file_name = 'trajectories_' + str(self.start_time) + '.npz'
        file_path = os.path.join(trajectories_dir, file_name)
        
        # Save trajectories using numpy npz format
        data_dict = {'trajectories': self.collected_data}
        np.savez(file_path, **data_dict)  # Save each trajectory as a key-value array
        rospy.loginfo("All atrajectories has been saved to file " + file_name)

    def check_user_input(self,):
        while not rospy.is_shutdown():
            self.enter_event.wait()
            with self.recording_lock:
                if not self.recording:
                    print('\n\n------------------------- [', len(self.collected_data), '] -------------------------')
                    input("Press ENTER to start new trajectory collection ...")
                    self.recording = True
                    rospy.loginfo("Collecting ...\n")
                    self.enter_event.clear()  # Sau khi nhấn ENTER, dừng chờ đến lần sau


if __name__ == '__main__':
    rospy.init_node('data_collector', anonymous=True)
    # Thresholds
    MIN_HEIGHT_THRESHOLD = 0.1  # Trajectory end height threshold
    MIN_FREQ_THRESHOLD = 100  # Minimum message frequency (Hz)
    MOCAP_OBJECT_TOPIC = '/mocap_pose_topic/frisbee1_pose'
    SWAP_YZ = False
    DECIMAL_NUM = 5

    collector = RoCatDataCollector(MOCAP_OBJECT_TOPIC, MIN_HEIGHT_THRESHOLD, MIN_FREQ_THRESHOLD, SWAP_YZ)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass