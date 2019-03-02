import numpy as np
import load_data
from map_utils import bresenham2D, mapCorrelation,texture_mapping
import matplotlib.pyplot as plt
from encoder import get_dynamic
import os
import cv2

#plot
def plot_without_revise(x,y):
    plt.plot(x,y)
    plt.title('trajectory')
    plt.xlabel('x, meter')
    plt.ylabel('y, meter')
    plt.show()
#time alignment
def time_align(target, time):
    num = len(target)
    idx = np.zeros(num, dtype=int)
    for i, v in enumerate(target):
        idx[i] = np.argmin(abs(v - time))
    return idx

# draw grid map
def mapping(grid, scan_lidar, curr_p, res, angle):
    empty_odds = np.log(0.9/0.1) / 4 #free space
    occup_odds = np.log(0.9/0.1) # occupied space
    saturated = 127
    # to cartesian
    dista = scan_lidar
    theta = angle + curr_p[2]
    x, y = dista * np.cos(theta), dista * np.sin(theta)
    # discretize
    x = (x / res).astype(int)
    y = (y / res).astype(int)


    empty_set = {}
    wall_set = {}
    for (a,b) in zip(x,y):
        line = np.array(bresenham2D(0, 0, a, b)).astype(int)
        xx = a + int(curr_p[0]) + grid.shape[0]//2
        yy = b + int(curr_p[1]) + grid.shape[1]//2
        wall_set[xx, yy] = True
        for j in range(len(line[0])-1):
            empty_set[(line[0][j], line[1][j])] = True
            
    for k, _ in wall_set.items():
        xx, yy = k[0], k[1]
        if 0 <= xx < grid.shape[0] and 0 <= yy < grid.shape[1]:
            grid[xx, yy] += occup_odds

    for k, _ in empty_set.items():
        xx = k[0] + int(curr_p[0]) + grid.shape[0]//2
        yy = k[1] + int(curr_p[1]) + grid.shape[1]//2
        if 0 <= xx < grid.shape[0] and 0 <= yy < grid.shape[1]:
            grid[xx, yy] -= empty_odds

    grid[grid > saturated] = saturated
    grid[grid < -saturated] = -saturated

    return grid
#resample
def stratified_resample(W):
    N = len(W)
    #random
    beta = (np.random.rand(N) + range(N)) / N

    w = np.cumsum(W)
    ret = []
    i, j = 0, 0
    while i < N:
        if beta[i] < w[j]:
            ret.append(j)
            i += 1
        else:
            j += 1
    return np.array(ret)

def prediction(grid, temp, X,angles, lidar_range, x_image, y_image, xs, ys, res):
    result = []
    for j in range(len(X)):
        world_angle = angles + X[j][2]
        #valid = np.logical_and(this_lidar >=0.1, this_lidar <=30)
        dista = lidar_range
        theta = world_angle
        x,y = dista*np.cos(theta), dista*np.sin(theta)
        x,y = x/res + grid.shape[0]//2, y/res + grid.shape[1]//2
        cor = mapCorrelation(temp, x_image, y_image, np.vstack((x,y)), (X[j][0] + xs)/res,(X[j][1] + ys)/res)
        #save best particle
        result.append(np.max(cor))
    return result
def update(now, cors, weight, particles):
    cors = weight * np.array(cors)
    e_x = np.exp(cors - np.max(cors))
    weight = e_x / e_x.sum()
    best = np.where(weight == np.max(weight))[0][0]
        
    now = particles[best].copy()
    now = now.ravel()
    return now
            
def draw_map(dataset, input_name, path, flag):
    #--load data
    lidar_angle_min, lidar_angle_max, lidar_angle_increment,\
    lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamsp = load_data.load_hokuyo(dataset)
    encoder_counts, encoder_stamps = load_data.load_encoder(dataset)
    imu_angular_velocity, imu_linear_acceleration, imu_stamps = load_data.load_imu(dataset)
    #if testing set, don't run image
    if dataset != 23 and flag == True:
        disp_stamps, rgb_stamps = load_data.load_kinect(dataset)
    
    #--time stamps align
    imu_idx = time_align(lidar_stamsp,imu_stamps)
    imu_stamps = imu_stamps[imu_idx]
    imu_angular_velocity = imu_angular_velocity[2, imu_idx]
    encoder_idx = time_align(lidar_stamsp, encoder_stamps)
    encoder_stamps = encoder_stamps[encoder_idx]
    encoder_counts = encoder_counts[:,encoder_idx]
    
    #--get current position
    x_list, y_list, theta_list,check = get_dynamic(dataset)
    pose = np.vstack((x_list,y_list,theta_list)).T
    
    #--map configuration
    res = 0.1
    grid_size = int(80/res)
    texture_size = int(80/res)
    
    grid = np.zeros((grid_size, grid_size))
    texture_map = np.zeros((texture_size, texture_size, 3))
    slam_map = np.zeros((grid_size, grid_size))
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    noise = np.array([0.015, 0.015, 0.08 * np.pi / 180])
    
    #--generate particle
    N = 200 #numbers of particle
    #N = 1 #test particle
    X = np.zeros((N,3)) #initial particle
    W = np.ones(N) / N #initial weight
    
    now = np.array(([0,0,0])) # initial position
    grid = mapping(grid, lidar_ranges[:,0], now, res, angles)
    pre_pose = pose[0]
    
    #--video
    video = cv2.VideoWriter(path+'video/grid_'+input_name+str(dataset)+'.avi', -1, 1, (grid_size, grid_size), isColor=3)
    if dataset != 23 and flag == True:
        video_t = cv2.VideoWriter(path+'video/texture_'+input_name+str(dataset)+'.avi', -1, 1, (texture_size, texture_size), isColor=3)
    
    
    for i in range(0,len(lidar_ranges[::1].T),100):
        if dataset != 23 and flag == True:
            #time align 
            tmp1_idx = np.argmin(np.abs(lidar_stamsp[i] - disp_stamps))
            tmp2_idx = np.argmin(np.abs(lidar_stamsp[i] - rgb_stamps))
            #get image name
            depth_img_name = 'disparity'+str(dataset) + '_' + str(tmp1_idx+1) + '.png'    
            rgb_img_name = 'rgb' + str(dataset) + '_' + str(tmp2_idx+1) + '.png'
        #get particle X
        delta = pose[i] - pre_pose
        noises = np.random.randn(N, 3) * noise
        #noises = 0
        X += delta + noises
        #X = pose[i] + noises
        X[:,2] %= 2*np.pi
        
        cors = []
        x_image, y_image = np.arange(grid.shape[0]), np.arange(grid.shape[1])
        l = 1
        xs, ys = np.arange(-res*l, res*l+res, res), np.arange(-res*l, res*l+res, res)
        temp = np.zeros_like(grid)
        temp[grid>0] = 1
        temp[grid<0] = -1
        #for each particle
        cors = prediction(grid,temp, X, angles, lidar_ranges[:,i],x_image, y_image,xs, ys, res)
        #update weight with softmax
        now = update(now, cors, W, X)
        f_now = now.copy()
        now[0] /= res
        now[1] /= res
        
        grid = mapping(grid,lidar_ranges[:,i],now,res,angles)
        #--create video
        grid_video = grid.copy()
        #--change grid color
        grid_video[grid_video > 0] = 10
        grid_video[grid_video < 0] = 1
        tmp_grid = np.zeros((grid_size, grid_size, 3))
        tmp_grid[:,:,0] = grid_video * 127
        tmp_grid[:,:,1] = grid_video * 127
        tmp_grid[:,:,2] = grid_video * 127
        cv2.circle(tmp_grid,(int(now[1]-3 + grid_size//2),int(now[0] + grid_size//2)),3,(0,0,255),-1)
        
        video.write(tmp_grid.astype(np.uint8))
        if dataset != 23 and flag == True:
    
            texture_map = texture_mapping(texture_map,f_now,os.path.join("dataRGBD\RGB" + str(dataset), rgb_img_name),os.path.join("dataRGBD\Disparity"+str(dataset), depth_img_name),res)
            texture_map_tmp = texture_map.copy()
            texture_map_tmp = cv2.cvtColor(texture_map_tmp.astype(np.uint8),cv2.COLOR_RGB2BGR)
            cv2.circle(texture_map_tmp,(int(now[1] + grid_size//2),int(now[0] + grid_size//2)),3,(0,0,255),-1)
            
            video_t.write(texture_map_tmp)
    
        #--check effective numbers of weight 
        n_eff = 1/(W**2).sum()
        #-- for debug
        if i % 500 == 0:
            print(X[0])
            print("i = ",i)
            print("now:",now)
            print("encoder:",pose[i]*10)
            print("n_eff:",n_eff)
        #--resample
        if n_eff < 0.8 * N:
            idx = stratified_resample(W)
            X[:] = X[idx]
            W.fill(1.0 / N)
        slam_map[int(now[0]) + slam_map.shape[0]//2, int(now[1])-3 + slam_map.shape[1]//2] = 1
        pre_pose = pose[i]
    #--change grid color
    grid[grid>0] = 2
    grid[np.logical_and(0<=grid, grid<=0)] = 0
    grid[grid<0] = 0.5
    
    #plot image
    slam_pos = np.where(slam_map == 1)
    grid_new = grid.copy()
    #draw trajectory
    for i in range(len(slam_pos[0])):
        cv2.circle(grid_new,(slam_pos[1][i],slam_pos[0][i]),3,(0,0,255),-1)
    #draw grid
    plt.figure()
    plt.axis('off')
    plt.imshow(grid_new, cmap='gray')
    plt.savefig(path + 'image/'+input_name+str(dataset)+'_grid_new.png')
    #if not testing data, draw texture map
    if dataset != 23 and flag == True:
        plt.figure()
        plt.axis('off')
        plt.imshow(texture_map)
        plt.savefig(path + 'image/'+input_name+str(dataset)+'_texture.png')
        video_t.release()
    video.release()

def main():
    dataset = 21
    input_name = 'train'
    path = 'C:/Users/TonyY/Desktop/All stuff/github/Simultaneous-Localization-and-Mapping/'
    flag = True
    draw_map(dataset, input_name, path, flag)

if __name__ == '__main__':
    main()

