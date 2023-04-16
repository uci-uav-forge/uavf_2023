# obstacle avoidance, updated for 3d and vectorized
import numpy as np
import random
from time import time


def obstacle_avoidance(centroids, dimensions):
  R = 16     # m
  angle = 43 # deg
  path_distance = 1
  drone_distance = 1
  increment = 3

  cur_heading = 0
  obstacle_avoidance = False

  #add a z boundry
  z_dist = 1 #m
  #check if any z coordinates are above the boundry and also chek if half the z dimension is intersecting the boundry
  c_new = []
  d_new = []
  for coords, dims in zip(centroids[:], dimensions[:]):
    if(np.abs(coords[2]) < z_dist or np.abs(coords[2])- dims[2]/2 < z_dist):
      c_new.append(coords)
      d_new.append(dims)
  centroids = np.array(c_new)
  dimensions = np.array(d_new)
  #remove that object from the obstacle list

  try:
    radius_list = np.amax(dimensions, axis=1) #list of all the radius's, max of all the dimensions
  except np.AxisError:
    return False

  distance_list = np.linalg.norm(centroids, axis=1) #distance of all the objects
  path_dist = np.abs(centroids[:,0]) #list of all the abs(x) values
  if(np.any(path_dist< radius_list+path_distance) or np.any(distance_list < radius_list + drone_distance)):
    obstacle_avoidance = True
  
  while obstacle_avoidance and cur_heading < angle:
    cur_heading += increment
    obstacle_avoidance = False

    radius_list = np.amax(dimensions, axis=1) #list of all the radius's
    distance_list = np.linalg.norm(centroids, axis=1) #distance of all the objects
    beta_list = np.arctan2(centroids[:,0], centroids[:,1]) - np.radians(cur_heading)
    path_dist = np.abs(distance_list * np.sin(beta_list))
    if(np.any(path_dist< radius_list+path_distance) or np.any(distance_list < radius_list + drone_distance)):
      obstacle_avoidance = True
    
  if(cur_heading > angle):
    return angle
  else:
    return cur_heading


def main(centers, dimensions):
  drone_heading = obstacle_avoidance(centers, dimensions)
  print(drone_heading)


def function3():
    centers = []
    dimensions = []
    for i in range(10):
        a = random.randint(-30, 30)
        b = random.randint(0, 15)
        f = random.randint(-30, 30)
        centers.append([a, b, f])
        c = random.random()
        d = random.random()
        e = random.random()
        dimensions.append([c,d,e])
    print(centers)
    print(dimensions)
    st = time()
    main(np.array(centers), np.array(dimensions))
    et= time()
    print(et-st)