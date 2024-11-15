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
from nae.utils.submodules.preprocess_utils.data_raw_reader import RoCatDataRawReader
from nae.utils.submodules.preprocess_utils.data_raw_correction_checker import RoCatDataRawCorrectionChecker

DEBUG = False
np.set_printoptions(suppress=True)


# main
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # Create a path to the directory ../trajectories
    file_path = os.path.join(parent_dir, 'data/bumerang1/min_len_65', '15-11-2024_17-57-59-traj_num-3.npz')

    data_reader = RoCatDataRawReader(file_path)
    data_collection_checker = RoCatDataRawCorrectionChecker()
    data_collection_checker.check_data_correction(data_reader.read())

    one_trajectory = data_reader.read()[1]
    print('check 111: ', one_trajectory['points'].shape)
    print('check 222: ', one_trajectory['msg_ids'].shape)
    print('check 333: ', one_trajectory['time_stamps'].shape)
    print('check 444: low_freq_num value ', one_trajectory['low_freq_num'])

    input('1')
    data_reader.check_data_correction()