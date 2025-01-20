import os
import pandas as pd
import numpy as np
import re
from python_utils.plotter import Plotter
global_plotter = Plotter()

def filter_csv_files(folder_path, keyword):
    """
    Filter all .csv files in the folder that match the format "ring_frisbee_<number>.csv".

    Args:
        folder_path (str): The path to the folder containing the files.

    Returns:
        list: A list of paths to the matching .csv files.
    """
    # Tạo regex pattern cho tên file
    # Tạo regex pattern động dựa trên keyword
    pattern = re.compile(rf'^{re.escape(keyword)}_\d+\.csv$')
    matching_files = []
    
    # Duyệt qua các file trong folder
    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):  # Kiểm tra tên file khớp pattern
            matching_files.append(os.path.join(folder_path, file_name))
    
    return matching_files

def read_csv_files(file_paths):
    """
    Read data from the given CSV file paths and store them in a list of dictionaries.

    Args:
        file_paths (list): List of paths to the CSV files.

    Returns:
        list: A list of dictionaries, each containing 'time_step' and 'position'.
    """
    if len(file_paths) == 0:
        raise ValueError("No CSV files found in the folder.")
    data_list = []
    for traj_idx, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] != 4:
                raise ValueError(f"Unexpected number of columns in file {file_path}.")

            data_dict = {
                'time_step': df[0].tolist(),
                'position': list(zip(df[1], df[2], df[3]))
            }
            data_list.append(data_dict)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return data_list

def save_data(data, object_name):
    # get current path
    current_path = os.path.dirname(os.path.realpath(__file__))
    # get parent path
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    output_dir = os.path.join(parent_path, 'data_preprocessed', object_name, f'{object_name}_{len(data)}_npz')
    # create folder if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, traj in enumerate(data):
        # Save as .npz instead of .npy
        output_file = os.path.join(output_dir, f'{object_name}_{idx}.npz')
        # Save the whole trajectory dictionary in .npz format
        np.savez(output_file, **traj)
    print(f'Saved processed data to {output_dir}')

def main():
    csv_folder = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data/water_bottle/2-position-treatment/water_bottle'
    object_name = 'water_bottle'
    
    csv_files = filter_csv_files(csv_folder, keyword=object_name)
    data = read_csv_files(csv_files)
    save_data(data, object_name)

if __name__ == '__main__':
    main()

