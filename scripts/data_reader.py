#!/usr/bin/env python3

'''
Program description:
    - Read npz file and plot the trajectory of the object.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
from python_utils.printer import Printer
DEBUG = False
np.set_printoptions(suppress=True)

class RoCatDataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories = np.load(self.file_path, allow_pickle=True)
        print('Loaded file with ', len(self.trajectories['trajectories']), ' trajectories')

        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Trajectory of the object')
        # self.ax.set_box_aspect([1,1,1])
        self.ax.view_init(elev=20, azim=30)

        self.util_printer = Printer()

    def plot_trajectories(self):
        traj_num = len(self.trajectories)
        for i in range(traj_num):
            trajectory = self.trajectories[i]
            points = trajectory['points']
            low_freq_num = trajectory[1]
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            z = [point[2] for point in points]
            self.ax.plot(x, y, z, color=self.colors[i], label=f'Trajectory {i+1}')
        plt.legend()
        plt.show() 
    
    def read(self):
        return self.trajectories['trajectories']
    
    '''
    check if the interpolation is correct
    Get 10 random trajectories from the trajectory and check if the velocity is correct
    '''
    def check_data_correction(self, swap_yz=False):
        # check swap_yz: If the data is swapped y and z, g should be the 8th element in one point
        # get 10 random trajectories

        all_data = self.read()
        sample_size = 10
        if len(all_data) < 10:
            print('Checking less data ({len(all_data)}) than expected')
            sample_size = len(all_data)
        
        indices = random.sample(range(len(all_data)), sample_size)

        count = 0
        for i in indices:
            # print a random point to check
            # print in blue background
            print('\n')
            self.util_printer.print_blue(f'{count} ----- checking trajectory {i} with {len(all_data[i]["points"])} points -----', background=True)
            count += 1
            point_random_idx = random.randint(0, len(all_data[i]) - 1)
            print(f'Trajectory {i} -> point {point_random_idx}:')
            print('     - Point:     ', all_data[i]['points'][point_random_idx])
            print('     - Timestamp: ', all_data[i]['time_stamps'][point_random_idx])
            result_swap_yz, msg = self.check_swap_yz_correction(all_data[i], swap_yz)
            if not result_swap_yz:
                # print in red
                self.util_printer.print_red(f'[SWAP YZ] Trajectory {i} has incorrect data')
                print('     ', msg)
                return False
            # print in green
            self.util_printer.print_green(f'[SWAP YZ] Trajectory {i} has correct data')
        
            if not self.check_velocity_correction(all_data[i]):
                print(f'\033[91m[VEL INTERPOLATION] Trajectory {i} has incorrect data')
                return False
            # print in green
            self.util_printer.print_green(f'[VEL INTERPOLATION] Trajectory {i} has correct data')
        return True
        
            
    
    # if swap y and z, the 9th element should be -9.81
    # else, the 8th element should be -9.81
    def check_swap_yz_correction(self, trajectory, swap_yz):
        for point in trajectory['points']:
            if swap_yz:
                value = point[8]
            else:
                value = point[7]
            if abs(value + 9.81) > 1e-5:
                msg = point
                return False, msg
        return True, ''
    def check_velocity_correction(self, trajectory):
        """
        Kiểm tra xem nội suy vận tốc có đúng không.
        The proper velocities vx, vy, vz should follow: 
        - Forward difference formula for the first point
        - Backward difference formula for the last point
        - Central difference formula for the rest of the points
        """
        traj_len = len(trajectory['points'])
        for i in range(traj_len):
            point = trajectory['points'][i]
            timestamp = trajectory['time_stamps'][i]
            if i == 0:
                # Forward difference
                next_point = trajectory['points'][i + 1]
                next_timestamp = trajectory['time_stamps'][i + 1]
                dt = next_timestamp - timestamp

                vx_expected = (next_point[0] - point[0]) / dt
                vy_expected = (next_point[1] - point[1]) / dt
                vz_expected = (next_point[2] - point[2]) / dt
            elif i == traj_len - 1:
                # Backward difference
                prev_point = trajectory['points'][i - 1]
                prev_timestamp = trajectory['time_stamps'][i - 1]
                dt = timestamp - prev_timestamp

                vx_expected = (point[0] - prev_point[0]) / dt
                vy_expected = (point[1] - prev_point[1]) / dt
                vz_expected = (point[2] - prev_point[2]) / dt
            else:
                # Central difference
                prev_point = trajectory['points'][i - 1]
                next_point = trajectory['points'][i + 1]
                prev_timestamp = trajectory['time_stamps'][i - 1]
                next_timestamp = trajectory['time_stamps'][i + 1]
                dt = next_timestamp - prev_timestamp

                vx_expected = (next_point[0] - prev_point[0]) / dt
                vy_expected = (next_point[1] - prev_point[1]) / dt
                vz_expected = (next_point[2] - prev_point[2]) / dt

            # Lấy vận tốc thực tế từ dữ liệu
            vx, vy, vz = point[3], point[4], point[5]

            # Kiểm tra nếu vận tốc khớp với sai số cho phép
            tolerance = 1e-5
            result = abs(vx - vx_expected) < tolerance and \
                    abs(vy - vy_expected) < tolerance and \
                    abs(vz - vz_expected) < tolerance
            if not result:
                print(f'\033[91m[VEL INTERPOLATION] Point {i} has incorrect velocity')
                print(f'Expected: vx = {vx_expected}, vy = {vy_expected}, vz = {vz_expected}')
                if i == 0:
                    print('Forward difference')
                    print('     ', trajectory['points'][i][:6])
                    print('         ', trajectory['time_stamps'][i])
                    print('     ', trajectory['points'][i + 1][:6])
                    print('         ', trajectory['time_stamps'][i + 1])
                elif i == traj_len - 1:
                    print('Backward difference')
                    print('     ', trajectory['points'][i - 1][:6])
                    print('         ', trajectory['time_stamps'][i - 1])
                    print('     ', trajectory['points'][i][:6])
                    print('         ', trajectory['time_stamps'][i])
                else:
                    print('Central difference')
                    print('     ', trajectory['points'][i - 1][:6])
                    print('         ', trajectory['time_stamps'][i-1])
                    print('     ', trajectory['points'][i][:6])
                    print('         ', trajectory['time_stamps'][i])
                    print('     ', trajectory['points'][i + 1][:6])
                    print('         ', trajectory['time_stamps'][i + 1])
                input()
            #print in green
            if i == 0:
                cal_way = 'Forward difference'
            elif i == traj_len - 1:
                cal_way = 'Backward difference'
            else:
                cal_way = 'Central difference'
            self.util_printer.print_green(f'[VEL INTERPOLATION] pass - {cal_way}', enable=DEBUG)
        return result
                
# main
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # Create a path to the directory ../trajectories
    file_path = os.path.join(parent_dir, 'data/bumerang1', '11-trajectories_09-11-2024_17-32-04.npz')

    data_reader = RoCatDataReader(file_path)
    # print(data_reader.trajectories.files)
    # input()
    # print(data_reader.trajectories['trajectories'])

    one_trajectory = data_reader.read()[1]
    print('check 111: ', one_trajectory['points'].shape)
    print('check 222: ', one_trajectory['msg_ids'].shape)
    print('check 333: ', one_trajectory['time_stamps'].shape)
    print('check 444: low_freq_num value ', one_trajectory['low_freq_num'])

    input('1')
    data_reader.check_data_correction()