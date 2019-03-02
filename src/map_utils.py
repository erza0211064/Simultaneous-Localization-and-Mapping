import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import imageio

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

def texture_mapping(texture_map,now,rgb_img_name,depth_img_name,res):
    #read rgb and depth image
    depth_img = imageio.imread(depth_img_name)
    rgb_img = imageio.imread(rgb_img_name)
    dd = -0.00304 * depth_img + 3.31
    depth = 1.03/dd
    #get rid of negative value and too large value
    idx = np.where(np.logical_and(depth > 0.01, depth < 5))
    #Intrinsic matrix parameter
    Intr_matrix = np.zeros((3,3))
    Intr_matrix[0,0] = 585.05108211
    Intr_matrix[0,1] = 0
    Intr_matrix[0,2] = 242.94140713
    Intr_matrix[1,1] = 585.05108211
    Intr_matrix[1,2] = 315.83800193
    Intr_matrix[2,2] = 1
    i = 0
    while i < len(idx[0]):        
        #caluate u,v from depth data
        rgb_i = (idx[0][i] * 526.37 + dd[idx[0][i],idx[1][i]] *(-4.5 *1750.46) + 19276.0)/585.051
        rgb_j = (idx[1][i] * 526.37 + 16662.0)/585.051
        
        pixels = np.array([rgb_i, rgb_j, 1]).T
        Zo = depth[idx[0][i],idx[1][i]]
        #from image frame to optical frame
        tmp_optical = Zo*np.linalg.inv(Intr_matrix).dot(pixels)
        optical_frame = np.hstack((tmp_optical,1))
        #ROC
        Roc = np.array([[0, -1.0,   0],
                        [0,  0,  -1.0],
                        [1.0,  0,   0]])
        
        # body to camera
        b_T_c = np.zeros((4,4))
        b_T_c[0,3] = 0.18
        b_T_c[1,3] = 0.005
        b_T_c[2,3] = 0.36
        b_T_c[3,3] = 1
        roll = 0
        pitch = 0.36
        yaw = 0.021
        Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                       [np.sin(yaw), np.cos(yaw),0],
                       [0          ,           0,1]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0            , 1,             0],
                       [-np.sin(pitch),0, np.cos(pitch)]])
        Rx = np.array([[1,             0,             0],
                       [0,  np.cos(roll), -np.sin(roll)],
                       [0,  np.sin(roll),  np.cos(roll)]])
        b_T_c[0:3,0:3] = (Rz.dot(Ry)).dot(Rx)
        
    
        # body to world
        w_T_b = np.array([[np.cos(now[2]), -np.sin(now[2]), 0, now[0]],
                          [np.sin(now[2]),  np.cos(now[2]), 0, now[1]],
                          [0             ,               0, 1, now[2]],
                          [0             ,               0, 0,      1]])
    
        # camera to body = camera to body body to world
        w_T_c = (w_T_b).dot(b_T_c)
        
    
        w_R_c = w_T_c[0:3,0:3]
        
        w_P_c = np.zeros((3,1))
        w_P_c[0] = w_T_c[0,3]
        w_P_c[1] = w_T_c[1,3]
        w_P_c[2] = w_T_c[2,3]
        
        ext_matrix = np.zeros((4,4))
        ext_matrix[0:3,0:3] = Roc.dot(w_R_c.T)
        ext_matrix[0:3,3:4] = -(Roc.dot(w_R_c.T)).dot(w_P_c)
        ext_matrix[3,3] = 1
        
        world = (np.linalg.inv(ext_matrix)).dot(optical_frame)
        #get x,y position
        xx = world[0]/res + texture_map.shape[0]//2
        yy = world[1]/res + texture_map.shape[1]//2
        prexx = xx
        preyy = yy
        #if z is small, draw texture map
        if world[2] < 2:    
            texture_map[int(xx),int(yy),:] = rgb_img[idx[0][i],idx[1][i],:]
            texture_map = texture_map.astype(int)
    
            
        
        if xx == prexx and yy == preyy:
            i+=100
        else:
            i += 10
    print("xx:",xx)
    print("yy:",yy)
    print("world:",world)

    
    
    return texture_map
    


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))
    

def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))

def test_mapCorrelation():
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("test_ranges.npy")

  # take valid indices
  indValid = np.logical_and((ranges < 30),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]

  # init MAP
  MAP = {}
  MAP['res']   = 0.05 #meters
  MAP['xmin']  = -20  #meters
  MAP['ymin']  = -20
  MAP['xmax']  =  20
  MAP['ymax']  =  20 
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
  

  
  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)
  
  # convert position in the map frame here 
  Y = np.stack((xs0,ys0))
  
  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
  
  # build an arbitrary map 
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1

  #import pdb
  #pdb.set_trace()
      
  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  x_range = np.arange(-0.2,0.2+0.05,0.05)
  y_range = np.arange(-0.2,0.2+0.05,0.05)


  
  print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
  ts = tic()
  c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
  toc(ts,"Map Correlation")

  c_ex = np.array([[3,4,8,162,270,132,18,1,0],
		  [25  ,1   ,8   ,201  ,307 ,109 ,5  ,1   ,3],
		  [314 ,198 ,91  ,263  ,366 ,73  ,5  ,6   ,6],
		  [130 ,267 ,360 ,660  ,606 ,87  ,17 ,15  ,9],
		  [17  ,28  ,95  ,618  ,668 ,370 ,271,136 ,30],
		  [9   ,10  ,64  ,404  ,229 ,90  ,205,308 ,323],
		  [5   ,16  ,101 ,360  ,152 ,5   ,1  ,24  ,102],
		  [7   ,30  ,131 ,309  ,105 ,8   ,4  ,4   ,2],
		  [16  ,55  ,138 ,274  ,75  ,11  ,6  ,6   ,3]])
    
  if np.sum(c==c_ex) == np.size(c_ex):
	  print("...Test passed.")
  else:
	  print("...Test failed. Close figures to continue tests.")	

  #plot original lidar points
  fig1 = plt.figure()
  plt.plot(xs0,ys0,'.k')

  #plot map
  fig2 = plt.figure()
  plt.imshow(MAP['map'],cmap="hot");

  #plot correlation
  fig3 = plt.figure()
  ax3 = fig3.gca(projection='3d')
  X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
  ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
  plt.show()
  
  
def show_lidar():
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("test_ranges.npy")
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(10)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()
	

if __name__ == '__main__':
  show_lidar()
  test_mapCorrelation()
  test_bresenham2D()
  
