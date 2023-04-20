# obstacle avoidance, updated for 3d and vectorized
import numpy as np
import math
import random
from time import time


def obstacle_avoidance(centroids, dimensions, max_hdg):
  R = 16     # m
  increment = 3  # deg

  path_distance = 1 # m
  drone_distance = 1 # m
  z_dist = 1 # m

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
    distance_list = np.linalg.norm(centroids, axis=1) #distance of all the objects
  except np.AxisError:
    return False
  
  right_hdg = 0
  left_hdg = 0
  
  # sweep right quadrant
  while right_hdg < max_hdg:
    beta_list = np.arctan2(centroids[:,0], centroids[:,1]) - np.radians(right_hdg)
    path_dist = np.abs(distance_list * np.sin(beta_list))
    if not (np.any(path_dist< radius_list+path_distance) or np.any(distance_list < radius_list + drone_distance)):
      break
    right_hdg += increment
  
  # sweep left quadrant
  while left_hdg > -max_hdg:
    beta_list = np.arctan2(centroids[:,0], centroids[:,1]) - np.radians(left_hdg)
    path_dist = np.abs(distance_list * np.sin(beta_list)) 
    if not (np.any(path_dist< radius_list+path_distance) or np.any(distance_list < radius_list + drone_distance)):
      break
    left_hdg -= increment
  
  # take the smallest heading deviation
  if abs(left_hdg) < right_hdg: return left_hdg
  else: return right_hdg


def main(centers, dimensions):
  drone_heading = obstacle_avoidance(centers, dimensions)
  print(drone_heading)


def function3():
  centers = []
  dimensions = []
  for i in range(60):
      a = random.randint(-30, 30)
      b = random.randint(0, 15)
      f = random.randint(-30, 30)
      centers.append([a, b, f])
      c = random.random()
      d = random.random()
      e = random.random()
      dimensions.append([c,d,e])
  #print(centers)
  #print(dimensions)
  st = time()
  main(np.array(centers), np.array(dimensions))
  et= time()
  print(et-st)


if __name__=='__main__':
  function3()