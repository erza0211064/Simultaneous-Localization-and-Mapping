#import sys
#sys.path.insert(0, 'C:/Users/TonyY/AppData/Local/Programs/Python/Python37-32/code/276HW2/data')
import load_data
import math
import numpy as np



def time_align(target, time):
    num = len(target)
    idx = np.zeros(num, dtype=int)
    for i, v in enumerate(target):
        idx[i] = np.argmin(abs(v - time))
    return idx

def calculate_encoder(a,b):
    return (a + b)*0.0022/2

def load_encoder_data(dataset):
    encoder_counts, encoder_stamps = load_data.load_encoder(dataset)
    imu_angular_velocity, imu_linear_acceleration, imu_stamps = load_data.load_imu(dataset)
    
    return encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps

def get_dynamic(dataset):
    #load data    
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_encoder_data(dataset)  
    lidar_angle_min, lidar_angle_max, lidar_angle_increment,\
    lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamsp = load_data.load_hokuyo(dataset)
    #align all the time with lidar
    imu_idx = time_align(lidar_stamsp,imu_stamps)
    imu_stamps = imu_stamps[imu_idx]
    imu_angular_velocity = imu_angular_velocity[2, imu_idx]
    encoder_idx = time_align(lidar_stamsp, encoder_stamps)
    encoder_stamps = encoder_stamps[encoder_idx]
    encoder_counts = encoder_counts[:,encoder_idx]
    
    w = imu_angular_velocity
    w = w[1:]
    #calculate the difference of time
    time = np.diff(encoder_stamps)
    
    FR = encoder_counts[0,1:]
    FL = encoder_counts[1,1:]
    RR = encoder_counts[2,1:]
    RL = encoder_counts[3,1:]
    
    left_wheel = ((FR + RR) / 2 * 0.0022).reshape(1,-1)
    left_wheel = np.hstack((np.zeros((1,1)),left_wheel))
    right_wheel = ((FL + RL) / 2 * 0.0022).reshape(1,-1)
    right_wheel = np.hstack((np.zeros((1,1)),right_wheel))
    robot_distance = (left_wheel + right_wheel) / 2
    d_theta = w*time
    theta = np.add.accumulate(d_theta)
    theta = np.hstack((0, theta))
    #x = np.add.accumulate(robot_distance * (np.sin(w/2)/w/2) * np.cos(theta) + w/2)
    #y = np.add.accumulate(robot_distance * (np.sin(w/2)/w/2) * np.sin(theta) + w/2)
    x = np.add.accumulate(robot_distance * np.cos(theta),axis=1)
    y = np.add.accumulate(robot_distance * np.sin(theta),axis=1)
    return x,y,theta.reshape((1,-1)), robot_distance
def get_velocity(dataset):
    #robotic configuration
    L = (0.47625 + 0.31115)/2
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_encoder_data(dataset)  
    FR = encoder_counts[0,:]
    FL = encoder_counts[1,:]
    RR = encoder_counts[2,:]
    RL = encoder_counts[3,:]
    left_wheel = []
    right_wheel = []
    robot_distance = []
    phi = []
    x = []
    y = []
    vl = []
    vr = []
    #omega = []
    velocity = []
    
    
    #initialize
    right_wheel.append(calculate_encoder(FR[0],RR[0]))
    left_wheel.append(calculate_encoder(FL[0],RL[0]))
    robot_distance.append((left_wheel[0]+right_wheel[0])/2)
    x = [0]
    y = [0]
    phi = [0]
    #calculate velocity
    for i in range(1,len(encoder_stamps)):
        right_wheel.append(calculate_encoder(FR[i],RR[i]))
        left_wheel.append(calculate_encoder(FL[i],RL[i]))
        robot_distance.append((left_wheel[i]+right_wheel[i])/2)
        
        phi.append(phi[i-1] + (right_wheel[i]-left_wheel[i])/L)
        x.append(x[i-1] + robot_distance[i]*math.cos(phi[i]))
        y.append(y[i-1] + robot_distance[i]*math.sin(phi[i]))
        
        vl.append((left_wheel[i])/(encoder_stamps[i] - encoder_stamps[i-1]))
        vr.append((right_wheel[i])/(encoder_stamps[i] - encoder_stamps[i-1]))
        #omega.append((vr[i-1] - vl[i-1])/L)
        #velocity.append((vr[i-1]+vl[i-1])/2)
        velocity.append(robot_distance[i-1]/(encoder_stamps[i] - encoder_stamps[i-1]))
    return velocity







