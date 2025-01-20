#!/usr/bin/env python3

'''
--------------------------------------------------------------------------------------------------------
Program description
    - Subscribes to a PoseStamped topic and records the trajectory of the object.
    - Start recording when the user presses ENTER 
    -   Interpolate velocity based on central difference method, not backward/forward difference method.
    -       I interpolate velocities manually instead of using np.gradient() because np.gradient() auto interpolates if delta t between time steps are not equal.
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
from python_utils.plotter import Plotter

class RoCatDataCollector:
    def __init__(self, mocap_object_topic,
            final_point_height_threshold,
            collection_area_x,
            collection_area_y,
            collection_area_z,
            low_freq_l1_threshold,
            low_freq_l2_threshold,
            low_freq_l2_num_threshold,
            gap_recog_threshold,
            min_len_traj, 
            missing_message_check):
        
        self.decima_prt_num = 5
        self.start_time = time.strftime("%d-%m-%Y_%H-%M-%S")     # get time now: d/m/y/h/m
        self.number_subscriber = rospy.Subscriber(mocap_object_topic, PoseStamped, self.pose_callback, queue_size=1)
        self.util_plotter = Plotter()

        self.mocap_object_topic = mocap_object_topic
        self.object_name = self.mocap_object_topic.split('/')[-1][:-5] # ignore _pose in the topic name
        self.final_point_height_threshold = final_point_height_threshold

        # Collecting area
        self.collection_area_x = collection_area_x
        self.collection_area_y = collection_area_y
        self.collection_area_z = collection_area_z
        
        # Low frequency treatment: Variables for checking message frequency
        self.low_freq_l1_threshold = low_freq_l1_threshold
        self.low_freq_l2_threshold = low_freq_l2_threshold
        self.low_freq_l2_num_threshold = low_freq_l2_num_threshold

        # Gap treatments
        self.gap_recog_threshold = gap_recog_threshold
        self.min_len_traj = min_len_traj
        self.missing_message_check = missing_message_check


        self.collected_data = []
        self.collected_data_raw = [] # no gap treatment data
        self.reset()
        self.thow_time_count = 0
        self.time_start = time.time()

        # Start user input thread
        self.enable_enter_check_event = threading.Event()
        self.enable_enter_check_event.set()  # Kích hoạt thread để kiểm tra ENTER
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

        ox = msg.pose.orientation.x
        oy = msg.pose.orientation.y
        oz = msg.pose.orientation.z
        ow = msg.pose.orientation.w

        self.current_position = [x, y, z]
        self.current_orientation = [ox, oy, oz, ow]
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
                                self.current_position[0] > self.collection_area_x[0] and self.current_position[0] < self.collection_area_x[1] and
                                self.current_position[2] > self.collection_area_z[0] and self.current_position[2] < self.collection_area_z[1])
            if enable_recording:    
                # check if messages are missing
                if self.missing_message_check==True:
                    if self.is_message_missing_signal(current_msg_id):
                        # end program
                        rospy.logerr("Reject the current trajectory ! Please recollect the trajectory !")
                        self.enable_enter_check_event.set()  # Kích hoạt thread để kiểm tra ENTER
                        self.reset()
                        # rospy.signal_shutdown("Node shutting down")
                        return
        
                # check if messages frequency is too low
                low_freq_level, freq = self.is_message_low_frequency_signal(current_timestamp, self.current_position)
                if low_freq_level == 2:
                    self.enable_enter_check_event.set()  # Kích hoạt thread để kiểm tra ENTER
                    self.reset()
                    rospy.logerr("Please recollect the trajectory !")
                    return          
                  
                # Save the current position to the current trajectory
                new_point = [self.current_position, self.current_orientation, current_timestamp, current_msg_id]
                self.current_trajectory.append(new_point)

                # print out
                new_point_prt = [
                    [round(val, self.decima_prt_num) for val in sublist] if isinstance(sublist, list) else round(sublist, self.decima_prt_num)
                    for sublist in new_point
                ]
                print('\n     ', len(self.current_trajectory)-1, ' - New point: pos = ', new_point_prt[0], ' ori = ', new_point_prt[1], ' time = ', new_point_prt[2], ' id = ', new_point_prt[3], ' low_freq_l1_count = ', self.low_freq_l1_count)
                if freq < self.low_freq_l2_threshold:
                    # print in purple color
                    print('\033[95m' + '             freq: ', freq, '\033[0m')
                else:
                    print('             freq: ', freq)
                print('             low_freq_level: ', low_freq_level)
                # calculate distance between 2 consecutive points
                if len(self.current_trajectory) > 1:    # backward
                    last_p = np.array(self.current_trajectory[-2][0][:3], dtype=float)
                    curr_p = np.array(self.current_trajectory[-1][0][:3], dtype=float)
                    distance = np.linalg.norm(curr_p - last_p)
                    if distance > 0.05 and distance < 0.8:
                        # print in yeallow color
                        print('\033[93m' + '             gap: ', distance, '\033[0m')
                    else:
                        print('             gap: ', distance)

                self.measure_callback_time(start_time)
                # Check end condition of the current trajectory
                if self.current_position[1] < self.final_point_height_threshold:
                    self.finalize_trajectory()
                    
            else:
                if len(self.current_trajectory) > self.min_len_traj:
                    # print in yellow color
                    print('\033[93m' + '     The object is out of the collection area. We need to stop adding point to current trajectory ealier than expected !' + '\033[0m')
                    self.finalize_trajectory()
                    self.reset()
                
        finally:
            # Release the lock
            self.recording_lock.release()

    def finalize_trajectory(self,):
        if self.process_data(self.current_trajectory, missing_message_check=self.missing_message_check):
            # Save all trajectories to file after each new proper trajectory
            enable_saving = input("     Do you want to save the current trajectory ? (y/n): ")
            while enable_saving not in ['y', 'n']:
                enable_saving = input("     Please input y or n: ")
            if enable_saving == 'y':
                self.save_trajectories_to_file()
                # print a point of the last trajectory
                random_point = self.collected_data[-1]['points'][9]
                # print in green color
                print('\n\033[92m' + '     Random point of the last trajectory: ' + str(random_point) + '\033[0m')
                print('         - Points: ', random_point)
                print('         - Orientation: ', self.collected_data[-1]['orientations'][9])
                print('         - Time: ', self.collected_data[-1]['time_stamps'][9])
                print('         - Low freq num: ', self.collected_data[-1]['low_freq_num'])
            else:
                rospy.logwarn("     The current trajectory is not saved !")
                # remove the last trajectory
                if self.collected_data:
                    self.collected_data.pop()
                if self.collected_data_raw:
                    self.collected_data_raw.pop()
        self.enable_enter_check_event.set()  # Kích hoạt thread để kiểm tra ENTER
        self.reset()     

    def reset(self,):
        self.current_position = [0, 0, 0]
        self.current_orientation = [0, 0, 0, 0]
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
    return: ID and freq
        ID:
            0 if frequency is fast enough
            1 if frequency is in range of low_freq_l1_threshold and low_freq_l2_threshold
            2 if frequency is lower than low_freq_l2_threshold
        freq: double
    '''
    def is_message_low_frequency_signal(self, current_timestamp, current_position):
        enable_low_freq_check = (self.last_timestamp != 0 and
                                 current_position[0] > self.collection_area_x[0] and current_position[0] < self.collection_area_x[1] and
                                 current_position[1] > self.collection_area_y[0] and current_position[1] < self.collection_area_y[1] and
                                 current_position[2] > self.collection_area_z[0] and current_position[2] < self.collection_area_z[1])
        if not enable_low_freq_check:
            self.last_timestamp = current_timestamp
            return 0, 0.0
        
        time_diff = (current_timestamp - self.last_timestamp)
        freq = 1.0 / time_diff
        self.last_timestamp = current_timestamp

        if freq >= self.low_freq_l1_threshold:
            return 0, freq
        if freq < self.low_freq_l1_threshold and freq >= self.low_freq_l2_threshold:
            freq = round(freq, 3)
            rospy.logwarn("Current message frequency is in warning level (100-110 hz): " + str(freq) + " Hz")
            self.low_freq_l1_count += 1
            return 1, freq
        if freq < self.low_freq_l2_threshold:
            if  self.low_freq_l2_count >= self.low_freq_l2_num_threshold:
                freq = round(freq, 3)
                rospy.logerr("Current message frequency is in err level (<100 hz): " + str(freq) + " Hz")
                self.low_freq_l1_count += 1
                self.low_freq_l2_count += 1
                rospy.logerr("The current trajectory is rejected due to low frequency level 2")
                return 2, freq
            return 1, freq
    
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
    return: point ID and distance
        Ex: 1, 2, 3, 4, 5, gap, 6, 7, 8, gap, 9, 10
        -> return [(6, distance), (9, distance)]
    '''
    def is_gap_trajectory(self, new_traj_np):
        gap_point_id = -1
        gap_dist = -1
        gaps = []
        # calculate distance between 2 consecutive points and load all gaps
        for i in range(1, len(new_traj_np)):    # backward
            curr_p = np.array(new_traj_np[i][:3], dtype=float)
            prev_p = np.array(new_traj_np[i-1][:3], dtype=float)
            dist = np.linalg.norm(curr_p - prev_p)
            if dist > self.gap_recog_threshold:
                gap_point_id = i
                gap_dist = dist
                gaps.append((gap_point_id, gap_dist))

        return gaps

    def gap_treatment(self, new_traj):
        name = '[GAP TREATMENT] '
        segments_good = []
        segments_bad = []

        # 1. count number of gaps
        pos_np = np.array([pose[0] for pose in new_traj])
        gaps = self.is_gap_trajectory(pos_np)

        # 2. return if there is no gap
        if len(gaps) == 0:
            # print in green color
            print('\033[92m' + name + 'There is no gap in the trajectory' + '\033[0m')
            if len(new_traj) < self.min_len_traj:
                rospy.logerr(name + "Gap error 2 - The trajectory is too short with " + str(len(new_traj)) + " points !")
                rospy.logerr(name + "      Please recollect the trajectory !")
                return False, None
            title = 'Trial ' + str(self.thow_time_count) + \
            ':\n     Data (' + str(len(self.collected_data)+1) + '): No gap' + \
            '\n     Raw data: ' + str(len(self.collected_data_raw)+1)
    
            print(' type new_traj: ', type(new_traj))
            input()
            # self.util_plotter.plot_samples_rviz([[new_traj, 'o']], title = title)
            return True, [new_traj]
        
        # 3. get segments_good based on gaps
        # print in yellow color
        print('\033[93m' + name + 'There are ' + str(len(gaps)) + ' gaps in the trajectory' + '\033[0m')

        # ori_np = np.array([pose[1] for pose in new_trajjj])
        # time_np = np.array([pose[2] for pose in new_trajjj])
        # id_np = np.array([pose[3] for pose in new_trajjj])
        last_gap_point_id = 0
        count = 0
        print('----- Segments -----')
        for gap in gaps:
            gap_point_id = gap[0]
            traj_seg = new_traj[last_gap_point_id : gap_point_id]
            # only keep the segment if it is long enough
            if gap_point_id - last_gap_point_id >= self.min_len_traj:
                segments_good.append(traj_seg)
                count += 1
            else:
                segments_bad.append(traj_seg)
                count += 1
            print('     Segment ', count, ' : ', len(traj_seg), ' points')
            last_gap_point_id = gap_point_id
        # add the last segment
        traj_seg = new_traj[last_gap_point_id:]
        if len(traj_seg) >= self.min_len_traj:
            segments_good.append(traj_seg)
        else:
            segments_bad.append(traj_seg)
        count += 1
        print('     Segment ', count, ' : ', len(traj_seg), ' points')
        print('There are ', len(segments_good), ' good segments and ', len(segments_bad), ' bad segments')
        
        # 4. plot -> only plot position data
        segments_good_plot = []
        segments_bad_plot = []
        segments_good_plot = [[seg[0], 'o'] for seg in segments_good]       # seg[0] is position seg[1] is orientation
        segments_bad_plot = [[seg[0], 'x'] for seg in segments_bad]

        # merge segments_bad to segments_good into 1 list
        segments_plot = segments_good_plot + segments_bad_plot


        # print('segments_plot: ', len(segments_plot))
        # print('segments_plot[0]: ', len(segments_plot[0]))
        # print('segments_plot[0][0]: ', segments_plot[0][0].shape)
        # input()




        title = 'Trial ' + str(self.thow_time_count) + \
            ':\n     Data: ' + str(len(self.collected_data)+1) + ' - ' + str(len(segments_plot)) +' segments' + \
            '\n     Raw data: ' + str(len(self.collected_data_raw)+1)
        # self.util_plotter.plot_samples_rviz(segments_plot, title = title)
        if len(segments_good) == 0:
            rospy.logerr(name + "Gap error 1 - No segment is long enough !")
            rospy.logerr(name + "      Please recollect the trajectory !")
            return False, None
    
        return True, segments_good

    def measure_callback_time(self, start_time):
        # measure callback function time
        end_time = rospy.Time.now().to_sec()
        callback_freq = 1.0 / (end_time - start_time)
        if callback_freq < self.low_freq_l2_threshold+30:
            rospy.logwarn("     Callback freq is low: " + str(round(callback_freq, 2)) + " Hz")

    # Process the current trajectory
    def process_data(self, new_trajectory, missing_message_check=True):
        # pos_np = np.array([pose[0] for pose in new_trajectory])
        # ori_np = np.array([pose[1] for pose in new_trajectory])
        # time_np = np.array([pose[2] for pose in new_trajectory])
        # id_np = np.array([pose[3] for pose in new_trajectory])
        traj_len = len(new_trajectory)
        if traj_len > 1:
            id_np = np.array([pose[3] for pose in new_trajectory])
            if missing_message_check:
                if self.is_missing_message_trajectory(id_np):
                    rospy.logerr("The data is not continuous, some messages from publisher might be missed\nplease recollect the trajectory !")
                    return False
            
            # gap treatment
            # print('new_traj_np shape: ', new_traj_np.shape[1]) # 5
            gt_result = self.gap_treatment(new_trajectory)
            if not gt_result[0]:
                return False
            segments = gt_result[1]
            for seg in segments:
                new_pro_seg = self.process_one_trajectory(seg)
                self.collected_data.append(new_pro_seg)
                # print in green color background
                pn = '\033[42m' + str(len(new_pro_seg['points'])) + ' points' + '\033[0m'
                print('\n     A new trajectory was collected with ' + pn)
            
            # process raw data (no gap treatment)
            new_pro_data_raw = self.process_one_trajectory(new_trajectory)
            self.collected_data_raw.append(new_pro_data_raw)
            # print in green color background
            pn_raw = '\033[42m' + str(len(new_pro_data_raw['points'])) + ' points' + '\033[0m'
            print('\n     A new RAW trajectory was collected with ' + pn_raw)

            return True
        else:
            rospy.logwarn("The trajectory needs to have at least 2 points to be saved !")
            rospy.logwarn("It seems that you haven't lifted the frisbee and thrown it yet !")
            rospy.logwarn("Please try again !")
            return False
    
    def process_one_trajectory(self, one_trajectory):
        pos_np = np.array([pose[0] for pose in one_trajectory])
        ori_np = np.array([pose[1] for pose in one_trajectory])
        time_np = np.array([pose[2] for pose in one_trajectory])

        vx = self.vel_interpolation(pos_np[:, 0], time_np)  # vx = dx/dt
        vy = self.vel_interpolation(pos_np[:, 1], time_np)  # vy = dy/dt
        vz = self.vel_interpolation(pos_np[:, 2], time_np)  # vz = dz/dt
        zeros = np.zeros_like(vx)
        gravity = np.full_like(vx, -9.81)

        extened_data_points = np.column_stack((pos_np[:, :3], vx, vy, vz, zeros, gravity, zeros))
        trajectory_data = {
            'points': extened_data_points,
            'orientations': ori_np,
            'msg_ids': id,
            'time_stamps': time_np,
            'low_freq_num': self.low_freq_l1_count
        }
        return trajectory_data
    
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
        # get object name from rostopic
        # Get the path of the current directory (where the script is running)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # Move one directory up
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        # --- 1. Save gap treated data ---
        # Create a path to the directory ../trajectories
        trajectories_dir = os.path.join(parent_dir, 'data', self.object_name, 'min_len_'+str(self.min_len_traj))
        # Create directory ../trajectories if it does not exist
        if not os.path.exists(trajectories_dir):
            os.makedirs(trajectories_dir)

        # # Delete old files containing 'trajectories_' in their name
        # old_files = glob.glob(os.path.join(trajectories_dir, '*' + str(self.start_time) + '-traj_num-' + '*'))
        # for old_file in old_files:
        #     try:
        #         os.remove(old_file)
        #         rospy.loginfo(f"Deleted old file: {old_file}")
        #     except OSError as e:
        #         rospy.logwarn(f"Error deleting file {old_file}: {e}")
        
        # Save trajectories using numpy npz format
        file_name = str(self.start_time) + '-traj_num-' + str(len(self.collected_data)) + '.npz'
        file_path = os.path.join(trajectories_dir, file_name)
        data_dict = {'trajectories': self.collected_data,
                    'object_name': self.mocap_object_topic}
        # np.savez(file_path, **data_dict)  # Save each trajectory as a key-value array
        # log_print = "[DATA] A new trajectory has been added and saved to file " + file_path
        # print in green color
        # print("\033[92m" + log_print + "\033[0m")

        # --- 2. Save raw data (no gap treatment) ---
        trajectories_raw_dir = os.path.join(parent_dir, 'data-no-gap-treatment', self.object_name, 'min_len_'+str(self.min_len_traj))
        if not os.path.exists(trajectories_raw_dir):
            os.makedirs(trajectories_raw_dir)
        # old_files = glob.glob(os.path.join(trajectories_raw_dir, '*' + str(self.start_time) + '-traj_num-' + '*'))
        # for old_file in old_files:
        #     try:
        #         os.remove(old_file)
        #         rospy.loginfo(f"Deleted old file: {old_file}")
        #     except OSError as e:
        #         rospy.logwarn(f"Error deleting file {old_file}: {e}")
        # Save trajectories using numpy npz format
        file_name = str(self.start_time) + '-traj_num-' + str(len(self.collected_data_raw)) + '.npz'
        file_path = os.path.join(trajectories_raw_dir, file_name)
        data_dict = {'trajectories': self.collected_data_raw}
        np.savez(file_path, **data_dict)  # Save each trajectory as a key-value array
        log_print = "[RAW DATA] A new trajectory has been added and saved to file " + file_path
        # print in green color
        print("\033[92m" + log_print + "\033[0m")


    def check_user_input(self,):
        while not rospy.is_shutdown():
            self.enable_enter_check_event.wait()
            with self.recording_lock:
                if not self.recording:
                    # print with blue background
                    time_collection = round((time.time() - self.time_start) / 60, 5)
                    print('\n\n' + '\033[44m' + '--------------------------------------------------------------------------------------------------------------' + '\033[0m')
                    # log_print = str(self.thow_time_count) + ' - ' + str(time_collection) + ' mins' + " -----------------------   Number of collected trajectories: [" + str(len(self.collected_data)) + "]   -----------------------"
                    # print("033[44m" + log_print + "\033[0m")
                    log_print = str(self.thow_time_count) + ' - ' + str(time_collection) + ' mins' + " ----------------------- Number of collected raw trajectories: [" + str(len(self.collected_data_raw)) + "] -----------------------"
                    print("033[44m" + log_print + "\033[0m")
                    print('                                         Press ENTER to start new trajectory collection')
                    # print in blue background
                    print('\033[44m' + '--------------------------------------------------------------------------------------------------------------' + '\033[0m')
                    input()

                    # self.util_plotter.reset_plot()
                    self.recording = True
                    # print in green color
                    print('\033[92m' + 'Ready ' + self.mocap_object_topic + '\033[0m')
                    self.thow_time_count += 1
                    self.enable_enter_check_event.clear()  # Sau khi nhấn ENTER, dừng chờ đến lần sau


if __name__ == '__main__':
    rospy.init_node('data_collector', anonymous=True)
    OBJECT_NAME = 'chocochips'
    FINAL_POINT_HEIGHT_THRESHOLD = 0.4  # The height of the final point of the trajectory to stop collecting a trajectory
    MISSING_MESSAGE_CHECK = False
    mocap_object_topic = f'/mocap_pose_topic/{OBJECT_NAME}_pose'  # The topic to subscribe to
    # Limit collection area
    collection_area = {
        'collection_area_x': [-1, 4],
        'collection_area_y': [0.0, 1000],
        'collection_area_z': [-2.0, 4.0]
    }
    
    # Low frequency treatment: Variables for checking message frequency
    low_freq_treatment_params = {
        'low_freq_l1_threshold': 110,  # Minimum message frequency (Hz)
        'low_freq_l2_threshold': 100,  # Minimum message frequency (Hz)
        'low_freq_l2_num_threshold': 3     # Number of times the message frequency is too low before applying low frequency level 2 treatment: reject current trajectory, reset to collect a new trajectory
    }

    # Gap treatments
    gap_treatment_params = {
        'gap_recog_threshold': 0.2,
        'min_len_traj': 50
    }


    collector = RoCatDataCollector(
        mocap_object_topic, 
        FINAL_POINT_HEIGHT_THRESHOLD, 
        **collection_area,
        **low_freq_treatment_params,
        **gap_treatment_params,
        missing_message_check=MISSING_MESSAGE_CHECK
    )
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 