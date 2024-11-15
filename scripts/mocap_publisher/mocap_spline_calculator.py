import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import statistics
# import matplotlib.pyplot as plt
import ipdb

import tf





class Spline_Calculator():
    def __init__(self):
        pass

    def quaternion_to_euler(self,quaternion):
        """Convert Quaternion to Euler Angles

        quarternion: geometry_msgs/Quaternion
        euler: geometry_msgs/Vector3
        """
        e = tf.transformations.euler_from_quaternion(
            (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        )
        return list(e)
    
    def Smoothing_Spline_sample(self, state_buffer):
        """the function is to calculate the method of smoothing spline

        Args:
            state_buffer (list): the list of the position and time of the rigid body
        
        Returns:
            velocity (list): the list of the velocity of the rigid body
        """
        # 時間軸（x軸）
        time = [0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ]  # = np.arange(0, 2*np.pi, .5)

        # 時系列観測データ
        # np.sin(time)+np.random.randn(len(y))/20.
        # 三角関数＋乱数
        state_buffer = np.array([ 0.01060855,  0.49403123,  0.7794513 ,  0.98637471,  0.96506601,
                0.65524765,  0.24423349, -0.4020829 , -0.86418404, -1.04206452, -0.97197121, -0.78730514, -0.28842717])
        
        time2 = np.arange(0, 2*np.pi-.1, .25)

        # 以下の２種類のλ（平滑パラメータ，ｓオプション）で検証

        spl1 = UnivariateSpline(time, state_buffer, s=0.0001)
        spl2 = UnivariateSpline(time, state_buffer, s=0.2)

        # plt.plot(time, state_buffer, '.', markersize=15, c='orange')  # 時系列観測データy（三角関数＋乱数）
        # plt.plot(time2, spl1(time2), '.', markersize=5, c='C0')  # s=0.0001で平滑化した補間データ
        # plt.plot(time2, spl1(time2), lw = .5, c='C0')  # spl1() 補間関数カーブ
        # plt.plot(time2, spl2(time2), '.', markersize=5, c='C1')  # s=0.2で平滑化した補間データ
        # plt.plot(time2, spl2(time2), lw = .5, c='C1')  # spl2() 補間関数カーブ
        # plt.legend(['y: data', 'spl1 ($\u03bb$=0.0001)', 'spl1(time)', 'spl2 ($\u03bb$=0.2)', 'spl2(time)'])
        

    def Smoothing_Spline(self, state_buffer):
        # # 時間軸（x軸）
        #time = [0.]
        time =[1688787528.1399214, 1688787528.154183, 1688787528.1561308, 1688787528.1696978, 1688787528.172974,
              1688787528.1811624, 1688787528.189757, 1688787528.1978698, 1688787528.2066374, 1688787528.2147322]
        # for i in range(len(state_buffer)-1):
            # ipdb.set_trace()
            # time.append((state_buffer[i][1].secs+state_buffer[i][1].nsecs*10**-9)-(state_buffer[i+1][1].secs+state_buffer[i+1][1].nsec*10**-9))
            
            # time.append(state_buffer[i][1].nsecs*10**-9-state_buffer[i+1][1].nsec*10**-9)
        # TRY check the tiem code indetals
        
        
        time=[]
        for time_secs_nsec in state_buffer:
            time.append(time_secs_nsec[1].secs+time_secs_nsec[1].nsecs*10**-9)
        
            

        # 以下の２種類のλ（平滑パラメータ，ｓオプション）で検証
        # 3rd order spline
        # spl1 = UnivariateSpline(time, state_buffer_x[:10],k=3, s=0.0001)
        # spl2 = UnivariateSpline(time, state_buffer_x[:10],k=3, s=0.2)
        # d_spl2_3 = UnivariateSpline(time, state_buffer_x[:10], s=.2, k=3).derivative(n=1)
        
        posi_x=[]
        posi_y=[]

        # euler_roll=[]
        # euler_pitch=[]
        euler_yaw=[]

        # ori_x=[]
        #ipdb.set_trace()
        for state_buffer_info in state_buffer:
            #ipdb.set_trace()
            posi_x.append(state_buffer_info[0].position.x)
            posi_y.append(state_buffer_info[0].position.y)
            # euler_roll.append(self.quaternion_to_euler(state_buffer_info[0].orientation)[0])
            # euler_pitch.append(self.quaternion_to_euler(state_buffer_info[0].orientation)[1])
            euler_yaw.append(self.quaternion_to_euler(state_buffer_info[0].orientation)[2]*(-1))
            
            # orientation sample
            # ori_z.append(state_buffer_info[0].orientation.z)
            
            #note: the orbit treat quat shap is {w,x,y,z}
            
        # spl2 = UnivariateSpline(time, posi_x,k=3, s=0.2)
        x_posi_vel = UnivariateSpline(time, posi_x, s=0.002, k=3).derivative(n=1)
        y_posi_vel = UnivariateSpline(time, posi_y, s=0.002, k=3).derivative(n=1)
        yaw_ang_vel = UnivariateSpline(time, euler_yaw, s=0.002, k=3).derivative(n=1)
        
        # print(f"x axis celocity is{x_posi_vel(time)},y axis velocity is {y_posi_vel(time)},yaw axis angle velocity is{yaw_ang_vel(time)}")
        
        #x_y_yaw_vel=[statistics.mean(x_posi_vel(time)), statistics.mean(y_posi_vel(time)), statistics.mean(yaw_ang_vel(time))]
        
        # getting newest value
        x_y_yaw_vel=[x_posi_vel(time)[-1], y_posi_vel(time)[-1], yaw_ang_vel(time)[-1]]
        # print(x_y_yaw_vel)


        #ipdb.set_trace()
        # plt.plot(time, state_buffer, '.', markersize=15, c='orange')  # 時系列観測データy（三角関数＋乱数）
        # plt.plot(time2, spl1(time2), '.', markersize=5, c='C0')  # s=0.0001で平滑化した補間データ
        # plt.plot(time2, spl1(time2), lw = .5, c='C0')  # spl1() 補間関数カーブ
        # plt.plot(time2, spl2(time2), '.', markersize=5, c='C1')  # s=0.2で平滑化した補間データ
        # plt.plot(time2, spl2(time2), lw = .5, c='C1')  # spl2() 補間関数カーブ
        # plt.legend(['y: data', 'spl1 ($\u03bb$=0.0001)', 'spl1(time)', 'spl2 ($\u03bb$=0.2)', 'spl2(time)'])

        return x_y_yaw_vel

if __name__ == "__main__":
    Spline_Calculator().Smoothing_Spline(list(range(10)))


    # self.state_buffer[0][1].secs*10**9+self.state_buffer[0][1].nsecs
