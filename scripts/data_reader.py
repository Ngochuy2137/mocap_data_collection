#!/usr/bin/env python3

'''
Program description:
    - Read npz file and plot the trajectory of the object.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(suppress=True)

class RoCatDataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories = np.load(self.file_path, allow_pickle=True)
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Trajectory of the object')
        self.ax.set_box_aspect([1,1,1])
        self.ax.view_init(elev=20, azim=30)

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
    
# main
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # Create a path to the directory ../trajectories
    file_path = os.path.join(parent_dir, 'data', 'trajectories_14-10-2024_03-29-23.npz')

    data_reader = RoCatDataReader(file_path)
    print(data_reader.trajectories.files)
    # print(data_reader.trajectories['trajectories'])
    for p in data_reader.trajectories['trajectories'][0]['points']:
        p_str = np.array2string(p, formatter={'float_kind':lambda x: "%.5f" % x})
        print(p_str)
    # data_reader.plot_trajectories()