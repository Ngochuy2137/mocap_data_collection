import numpy as np
import pandas as pd
import os
from python_utils.plotter import Plotter
global_plotter = Plotter()

class NPZDataRawReader:
    def __init__(self, file_path):
        # load the npz file
        self.trajectories = np.load(file_path, allow_pickle=True)['trajectories']
        print('Loaded ', len(self.trajectories), ' trajectories')
    
    def read_raw_data(self):
        return self.trajectories
    
class DataFile2Excel:
    def __init__(self, data_file, object_name, data_owner='RLLAB'):
        self.object_name = object_name
        self.data_owner = data_owner
        self.data = NPZDataRawReader(data_file).read_raw_data()
        print('Loaded file with ', len(self.data), ' trajectories')

    def save_to_excel(self, need_to_swap_y_z_for_rllab_data=False, plot=False, outlier_list=[]):
        '''
        save each trajectory to an excel file
        '''
        data_folder = self.create_data_folder()
        i = 68
        for traj_idx, traj in enumerate(self.data):
            # if traj_idx == 2 or traj_idx == 26:
            #     print('Skip traj ', traj_idx); input()
            #     global_plotter.plot_trajectory_dataset_matplotlib(samples=[traj['points'][:, :3]], title=f'Trajectory {traj_idx}', rotate_data_whose_y_up=True)
            #     continue
            if plot:
                if len(outlier_list) > 0:
                    if traj_idx in outlier_list:
                        print('Skip traj ', traj_idx)
                        global_plotter.plot_trajectory_dataset_matplotlib(samples=[traj['points'][:, :3]], title=f'Trajectory {traj_idx}', rotate_data_whose_y_up=True)
                        continue
                else:
                    global_plotter.plot_trajectory_dataset_matplotlib(samples=[traj['points'][:, :3]], title=f'Trajectory {traj_idx}', rotate_data_whose_y_up=True)

            one_data_dict = {}
            i += 1

            # dict_keys(['points', 'msg_ids', 'time_stamps', 'low_freq_num'])
            for key, value in traj.items():
                if key == 'msg_ids':
                    one_data_dict['msg_ids'] = value
                elif key == 'time_stamps':
                    one_data_dict['time_stamps'] = value
                    # print('time_stamps: ', value); input()
                elif key == 'points':
                    one_data_dict['pos_x'] = [v[0] for v in value]
                    if need_to_swap_y_z_for_rllab_data:
                        one_data_dict['pos_y'] = [v[2] for v in value]
                        one_data_dict['pos_z'] = [v[1] for v in value]
                    else:
                        one_data_dict['pos_y'] = [v[1] for v in value]
                        one_data_dict['pos_z'] = [v[2] for v in value]
                    one_data_dict['vel_x'] = [v[3] for v in value]
                    if need_to_swap_y_z_for_rllab_data:
                        one_data_dict['vel_y'] = [v[5] for v in value]
                        one_data_dict['vel_z'] = [v[4] for v in value]
                    else:
                        one_data_dict['vel_y'] = [v[4] for v in value]
                        one_data_dict['vel_z'] = [v[5] for v in value]

            one_data_df = pd.DataFrame(one_data_dict)

            if self.data_owner == 'RLLAB':
                desired_order = ['msg_ids', 'time_stamps'] + [col for col in one_data_df.columns if col not in ['msg_ids', 'time_stamps']]
                one_data_df = one_data_df[desired_order]
        
            output_file = os.path.join(data_folder, f'{self.object_name}_{i}.xlsx')
            one_data_df.to_excel(output_file, index=False)
            print('Saved to ', output_file)

    def create_data_folder(self,):
        # get directory of this script
        data_folder = os.path.dirname(os.path.realpath(__file__))
        # cd ..
        data_folder = os.path.dirname(data_folder)
        # cd to data folder
        data_folder = os.path.join(data_folder, 'data', self.data_owner+'_dataset_excel', f'{self.object_name}_excel')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        return data_folder

# main
def main():
    ## ================= RLLAB dataset =================
    data_dir = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/mocap_ws/src/mocap_data_collection/data-no-gap-treatment/chocochips/min_len_50/20-01-2025_10-48-35-traj_num-41.npz'
    object_name = 'cookie_box'
    PLOT = True
    # OUTLIERS = [52, 62, 71, 76, 89]
    # OUTLIERS = [3, 17, 47, 64, 65, 66, 93, 97, 102, 110]
    OUTLIERS = [64, 68]

    data_to_excel = DataFile2Excel(data_dir,object_name=object_name)
    '''
    need_to_swap_y_z_for_rllab_data: when colleting data, we swapped y and z axis for rllab data, however, after that, we decided to keep the original data format with y axis as up axis
    so we need to swap y and z axis back to the original format by setting need_to_swap_y_z_for_rllab_data=True
    '''
    data_to_excel.save_to_excel(need_to_swap_y_z_for_rllab_data=False, plot=PLOT, outlier_list=OUTLIERS)

if __name__ == '__main__':
    main()
