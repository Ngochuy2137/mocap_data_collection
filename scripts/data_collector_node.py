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
            - We can know how many messages are missed by checking topic field /topic/header/seq (read self.last_msg_id variable) (Level 1)
                - Message missing check -level 2: Before process a new trajectory, we also check if data is uncontinuous (check self.is_uncontinuous_data(msg_ids))
                    - If data is uncontinuous, that means some messages are missing -> Check /topic/header/seq field

Maintainer: Huynn
--------------------------------------------------------------------------------------------------------
'''

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
import time
import threading
import os
import glob

class RoCatDataCollector:
    def __init__(self, mocap_object_topic,
                swap_yz,
                stop_traj_z_threshold,
                low_freq_level1_threshold, 
                low_freq_level2_threshold, 
                low_freq_l2_num_threshold, 
                low_freq_x_start_check, 
                low_freq_y_start_check, 
                low_freq_z_start_check,
                gap_threshold):
        
        self.decima_prt_num = 5
        self.start_time = time.strftime("%d-%m-%Y_%H-%M-%S")     # get time now: d/m/y/h/m
        self.number_subscriber = rospy.Subscriber(mocap_object_topic, PoseStamped, self.pose_callback, queue_size=200)
        self.mocap_object_topic = mocap_object_topic

        self.stop_traj_z_threshold, = stop_traj_z_threshold,

        self.low_freq_level1_threshold = low_freq_level1_threshold
        self.low_freq_level2_threshold = low_freq_level2_threshold
        # Conditions for apply low frequency level 2 treatment
        self.low_freq_l2_num_threshold = low_freq_l2_num_threshold
        self.low_freq_x_start_check = low_freq_x_start_check
        self.low_freq_y_start_check = low_freq_y_start_check
        self.low_freq_z_start_check = low_freq_z_start_check
        self.gap_threshold = gap_threshold

        self.swap_yz = swap_yz
        self.collected_data = []
        self.reset()
        self.thow_time_count = 0

        # Start user input thread
        self.enter_event = threading.Event()
        self.enter_event.set()  # Kích hoạt thread để kiểm tra ENTER
        self.recording_lock = threading.Lock()
        self.user_input_thread = threading.Thread(target=self.check_user_input)
        self.user_input_thread.daemon = True
        self.user_input_thread.start()

        self.real_last_msg_time = 0 # for debug

    def pose_callback(self, msg:PoseStamped):
        # measure callback function time
        start_time = rospy.Time.now().to_sec()
        # get x, y, z
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        # Swap y and z
        if self.swap_yz:
            y, z = z, y
        self.current_position = [x, y, z]
        current_timestamp = msg.header.stamp.to_sec()
        current_msg_id = msg.header.seq

        # # check if messages are missing
        # if self.is_message_missing_signal(current_msg_id):
        #     rospy.logwarn("Becareful !")
        #     return

        # Check if the lock is released (if user pressed ENTER), else: skip the callback
        if not self.recording_lock.acquire(blocking=False):
            # rospy.loginfo("Skipping message due to lock being held.")
            return
        '''
        try...finally ensures that the lock will always be released, no matter what happens during processing inside the try block. 
        Without finally, when an error (or exception) occurs, the lock can be held forever, and no other thread can obtain the lock.
        '''
        try:
            enable_recording = (self.recording and 
                                self.current_position[0] > self.low_freq_x_start_check and
                                self.current_position[1] > self.low_freq_y_start_check)
            if enable_recording:    
                # check if messages are missing
                if self.is_message_missing_signal(current_msg_id):
                    # end program
                    rospy.logerr("Reject the current trajectory ! Please recollect the trajectory !")
                    self.enter_event.set()  # Kích hoạt thread để kiểm tra ENTER
                    self.reset()
                    # rospy.signal_shutdown("Node shutting down")
                    return
        
                # check if messages frequency is too low
                low_freq_level = self.is_message_low_frequency_signal(current_timestamp, self.current_position)
                if low_freq_level == 2:
                    self.enter_event.set()  # Kích hoạt thread để kiểm tra ENTER
                    self.reset()
                    rospy.logerr("Please recollect the trajectory !")
                    return          
                  
                # Save the current position to the current trajectory
                new_point = self.current_position + [current_timestamp] + [current_msg_id]
                self.current_trajectory.append(new_point)

                # print out
                new_point_prt = [round(i, self.decima_prt_num) for i in new_point]
                print('     New point: x=', new_point_prt[0], ' y=', new_point_prt[1], ' z=', new_point_prt[2], ' t=', new_point_prt[3], ' low_freq_l1_count=', self.low_freq_l1_count)

                self.measure_callback_time(start_time)
                # Check end condition of the current trajectory
                if self.current_position[2] < self.stop_traj_z_threshold:
                    # rospy.loginfo("Trajectory ended with " + str(len(self.current_trajectory)) + " points !")
                    if self.process_trajectory(self.current_trajectory):
                        # Save all trajectories to file after each new proper trajectory
                        enable_saving = input("     Do you want to save the current trajectory ? (y/n): ")
                        if enable_saving == 'y':
                            self.save_trajectories_to_file()
                        else:
                            rospy.logwarn("     The current trajectory is not saved !")
                    self.enter_event.set()  # Kích hoạt thread để kiểm tra ENTER
                    self.reset()
                
        finally:
            # Release the lock
            self.recording_lock.release()
                    
    def reset(self,):
        self.current_position = [0, 0, 0]
        self.current_trajectory = []
        self.low_freq_l1_count = 0
        self.low_freq_l2_count = 0
        self.recording = False
        self.last_msg_id = 0
        self.last_timestamp = 0.0
    
    '''
    Check if messages are missing
    return:
        False: no message is missing
        True: messages are missing
    '''
    def is_message_missing_signal(self, current_msg_id):
        if self.last_msg_id == 0:
            self.last_msg_id = current_msg_id
            return False

        if current_msg_id - self.last_msg_id == 1:
            self.last_msg_id = current_msg_id
            return False
        elif current_msg_id - self.last_msg_id > 1:
            # list all missing messages
            miss_mess = [i for i in range(self.last_msg_id + 1, current_msg_id)]
            # rospy.logwarn("\n[----------------------------------------------- WARNING -----------------------------------------------]")
            rospy.logwarn("      Some messages missing: " + str(miss_mess))
            rospy.logwarn("      Please check the connection between the mocap system and the computer which runs this subscriber.")
            # rospy.logwarn("[---------------------------------------------------------------------------------------------------------]")
            self.last_msg_id = current_msg_id
            return True
        
        self.last_msg_id = current_msg_id
        return False
        
    '''
    Check if messages frequency is too low
    return:
        0 if frequency is fast enough
        1 if frequency is in range of low_freq_level1_threshold and low_freq_level2_threshold
        2 if frequency is lower than low_freq_level2_threshold
    '''
    def is_message_low_frequency_signal(self, current_timestamp, current_position):
        enable_low_freq_check = (self.last_timestamp != 0 and
                                 current_position[0] > self.low_freq_x_start_check and
                                 current_position[1] > self.low_freq_y_start_check and
                                 current_position[2] > self.low_freq_z_start_check)
        if not enable_low_freq_check:
            self.last_timestamp = current_timestamp
            return 0
        
        time_diff = (current_timestamp - self.last_timestamp)
        freq = 1.0 / time_diff
        if freq >= self.low_freq_level1_threshold:
            self.last_timestamp = current_timestamp
            return 0
        if freq < self.low_freq_level1_threshold and freq >= self.low_freq_level2_threshold:
            freq = round(freq, 3)
            rospy.logwarn("Current message frequency is in warning level (100-110 hz): " + str(freq) + " Hz")
            self.last_timestamp = current_timestamp
            self.low_freq_l1_count += 1
            return 1
        if freq < self.low_freq_level2_threshold:
            if  self.low_freq_l2_count >= self.low_freq_l2_num_threshold:
                freq = round(freq, 3)
                rospy.logerr("Current message frequency is in err level (<100 hz): " + str(freq) + " Hz")
                self.last_timestamp = current_timestamp
                self.low_freq_l1_count += 1
                self.low_freq_l2_count += 1
                rospy.logerr("The current trajectory is rejected due to low frequency level 2")
                return 2
            return 1
    
    '''
    Check if data missing after collecting a completed trajectory
    return:
        True if data missing
        False if data is continuous
    '''
    def is_missing_message_trajectory(self, msg_ids):
        # check msg_ids list is continuous or not
        for i in range(1, len(msg_ids)):
            if msg_ids[i] - msg_ids[i - 1] != 1:
                return True
        return False
    
    '''
    Check if a trajectory is split into multiple segments in case the mocap cannot track the object
    If the distance between 2 consecutive points is greater than a certain distance, the trajectory is considered uncontinuous
    return:
        -1 if data is uncontinuous
        >= 0 if data is continuous
    '''
    def is_uncontinuous_trajectory(self, msg_ids):
        # calculate distance between 2 consecutive points
        for i in range(1, len(msg_ids)):
            next_p = np.array(self.current_trajectory[i][:3], dtype=float)
            prev_p = np.array(self.current_trajectory[i - 1][:3], dtype=float)
            if np.linalg.norm(next_p - prev_p) > self.gap_threshold:
                return i
        return -1

    def measure_callback_time(self, start_time):
        # measure callback function time
        end_time = rospy.Time.now().to_sec()
        callback_freq = 1.0 / (end_time - start_time)
        if callback_freq < self.low_freq_level2_threshold+30:
            rospy.logwarn("     Callback freq is low: " + str(round(callback_freq, 2)) + " Hz")

    # Process the current trajectory
    def process_trajectory(self, new_trajectory):
        # Interpolate vx, vy, vz from x, y, z
        new_traj_np = np.array(new_trajectory)
        if len(new_traj_np) > 1:
            msg_ids = new_traj_np[:, 4]
            if self.is_missing_message_trajectory(msg_ids):
                rospy.logerr("The data is not continuous, some messages from publisher might be missed\nplease recollect the trajectory !")
                return False
            
            gap = self.is_uncontinuous_trajectory(msg_ids)
            if gap >2:
                rospy.logerr("There are some gaps in the trajectory, point: " + str(gap))
                rospy.logerr("please recollect the trajectory !")
                return False
            
            time_stamps = new_traj_np[:, 3]

            vx = self.vel_interpolation(new_traj_np[:, 0], time_stamps)  # vx = dx/dt
            vy = self.vel_interpolation(new_traj_np[:, 1], time_stamps)  # vy = dy/dt
            vz = self.vel_interpolation(new_traj_np[:, 2], time_stamps)  # vz = dz/dt
            zeros = np.zeros_like(vx)
            gravity = np.full_like(vx, 9.81)

            # print('new_traj_np[:, :3] shape: ', new_traj_np[:, :3].shape)
            # print('vx shape: ', vx.shape)
            # print('vy shape: ', vy.shape)
            # print('vz shape: ', vz.shape)
            # print('zeros shape: ', zeros.shape)
            # print('gravity shape: ', gravity.shape)
            print('\n     --------- A new trajectory was collected with ' + str(len(new_traj_np)) + ' points ---------')
            print('     -------------------------------------------------------------------')

            extened_data_points = np.column_stack((new_traj_np[:, :3], vx, vy, vz, zeros, zeros, gravity))
            # remove the segment with gap, if the gap point is 0 or 1 or 2
            extened_data_points = extened_data_points[:(gap+1)]
            trajectory_data = {
                'points': extened_data_points,
                'msg_ids': msg_ids,
                'time_stamps': time_stamps,
                'low_freq_num': self.low_freq_l1_count
            }
            self.collected_data.append(trajectory_data)
            return True
        else:
            rospy.logwarn("The trajectory needs to have at least 2 points to be saved !")
            rospy.logwarn("It seems that you haven't lifted the frisbee and thrown it yet !")
            rospy.logwarn("Please try again !")
            return False
    
    def vel_interpolation(self, x_arr, t_arr):
        vel = []
        for i in range(len(x_arr)):
            if i == 0:
                # Forward difference
                prev_t_id = i
                next_t_id = i+1
            elif i == len(x_arr) - 1:
                # Backward difference
                prev_t_id = i-1
                next_t_id = i
            else:
                # Central difference
                prev_t_id = i-1
                next_t_id = i+1
            
            #check 0 division
            if t_arr[next_t_id] - t_arr[prev_t_id] == 0:
                vel_i = 0
            else:
                vel_i = (x_arr[next_t_id] - x_arr[prev_t_id]) / (t_arr[next_t_id] - t_arr[prev_t_id])
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

        # Delete old files containing 'trajectories_' in their name
        old_files = glob.glob(os.path.join(trajectories_dir, '*trajectories_' + str(self.start_time) + '*'))
        for old_file in old_files:
            try:
                os.remove(old_file)
                rospy.loginfo(f"Deleted old file: {old_file}")
            except OSError as e:
                rospy.logwarn(f"Error deleting file {old_file}: {e}")
        
        # Save trajectories using numpy npz format
        file_name = str(len(self.collected_data)) + '-trajectories_' + str(self.start_time) + '.npz'
        file_path = os.path.join(trajectories_dir, file_name)
        data_dict = {'trajectories': self.collected_data}
        np.savez(file_path, **data_dict)  # Save each trajectory as a key-value array
        log_print = "A new trajectory has been added and saved to file " + file_name
        # print in green color
        print("\033[92m" + log_print + "\033[0m")

    def check_user_input(self,):
        while not rospy.is_shutdown():
            self.enter_event.wait()
            with self.recording_lock:
                if not self.recording:
                    # print with blue background
                    log_print = str(self.thow_time_count) + " ------------------------- Number of collected trajectories: [" + str(len(self.collected_data)) + "] -------------------------"
                    print("\n\n\n\n\n\033[44m" + log_print + "\033[0m")
                    log_print = "\033[44mPress ENTER to start new trajectory collection ... \033[0m"
                    input(log_print)
                    self.recording = True
                    rospy.loginfo("Collecting ...\n")
                    self.thow_time_count += 1
                    self.enter_event.clear()  # Sau khi nhấn ENTER, dừng chờ đến lần sau


if __name__ == '__main__':
    rospy.init_node('data_collector', anonymous=True)
    MOCAP_OBJECT_TOPIC = '/mocap_pose_topic/frisbee1_pose'
    # Variables for stop collecting current trajectory
    STOP_TRAJ_Z_THRESHOLD = 0.4  # Trajectory end height threshold

    # Variables for checking message frequency
    LOW_FREQ_LEVEL1_THRESHOLD = 110  # Minimum message frequency (Hz)
    LOW_FREQ_LEVEL2_THRESHOLD = 100  # Minimum message frequency (Hz)
    LOW_FREQ_L2_NUM_THRESHOLD = 3     # Number of times the message frequency is too low before applying low frequency level 2 treatment: reject current trajectory, reset to collect a new trajectory
    # Condition to activate the low frequency check 
    low_freq_check_conditions = {
        'low_freq_x_start_check': -0.5,     # Start to check when x > low_freq_x_start_check
        'low_freq_y_start_check': -1.4,     # Start to check when y > low_freq_y_start_check
        'low_freq_z_start_check': 0.4      # Start to check when z > low_freq_z_start_check
    }
    SWAP_YZ = True
    GAP_THRESHOLD = 0.05  # Gap threshold to check if the trajectory is uncontinuous

    collector = RoCatDataCollector(
        MOCAP_OBJECT_TOPIC, 
        SWAP_YZ, 
        STOP_TRAJ_Z_THRESHOLD, 
        LOW_FREQ_LEVEL1_THRESHOLD, 
        LOW_FREQ_LEVEL2_THRESHOLD, 
        LOW_FREQ_L2_NUM_THRESHOLD,
        low_freq_check_conditions['low_freq_x_start_check'], 
        low_freq_check_conditions['low_freq_y_start_check'], 
        low_freq_check_conditions['low_freq_z_start_check'],
        GAP_THRESHOLD
    )
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass