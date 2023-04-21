# obstacle avoidance, updated for 3d and vectorized
import numpy as np
import math
import random
from time import time
from numba import njit, prange


def path_is_safe(hdg, mag_arr, radius_arr, Xs, Ys, path_dist):
  phis = np.abs(hdg - (90 - np.degrees(np.arctan(Ys/Xs))))
  D_diff = mag_arr*(np.sin(np.radians(phis))) - radius_arr
  masked_D_diff = np.ma.masked_greater(D_diff, path_dist)
  bool_arr = np.ma.getmaskarray(masked_D_diff)
  return np.all(bool_arr)


def obstacle_avoidance(centroids, dimensions, max_hdg):
  increment = 5  # deg
  path_dist = 1 # m
  drone_dist = 1
  z_dist = 1 # m

  print(centroids)
  print()

  #check if any z coordinates are above the boundry and also check if half the z dimension is intersecting the boundry
  c_new = []
  d_new = []

  #for coords, dims in zip(centroids[:], dimensions[:]):
  for i in range(len(centroids)):
    if (centroids[i][2] >= 0 and centroids[i][2]-dimensions[i][2] < z_dist)\
    or (centroids[i][2] < 0 and centroids[i][2]+dimensions[i][2] > -z_dist):
      c_new.append(centroids[i])
      d_new.append(dimensions[i])

  centr_arr = np.array(c_new)
  dim_arr = np.array(d_new)
  print(centr_arr)

  # approximate max len dimension as sphereical radius
  # if no dangers present, skip obstacle avoidance
  try:
    radius_arr = np.amax(dim_arr, axis=1) 
  except np.AxisError:
    return False
  
  twoD_centroids = np.delete(centr_arr, 2, axis=1)
  mag_arr = np.zeros(len(twoD_centroids))
  for i in range(len(twoD_centroids)):
    mag_arr[i] = np.sqrt(twoD_centroids[i].dot(twoD_centroids[i]))
  
  x_centrs = twoD_centroids[:,0]
  y_centrs = twoD_centroids[:,1]
  right_hdg = 0
  left_hdg = 0

  # sweep right quadrant
  while right_hdg < max_hdg + 1:
    if path_is_safe(right_hdg, mag_arr, radius_arr, x_centrs, y_centrs, path_dist):
      break
    right_hdg += increment
  
  # sweep left quadrant
  while left_hdg > -max_hdg - 1:
    if path_is_safe(left_hdg, mag_arr, radius_arr, x_centrs, y_centrs, path_dist):
      break
    left_hdg -= increment

  print(left_hdg)
  print(right_hdg)
  if abs(left_hdg) < right_hdg: return left_hdg
  else: return right_hdg


if __name__=='__main__':
  #centr_arr = np.array([[2, 16, 0]] )
  #dim_arr = np.array([[8, 8, 8]])
  n=50
  centr_arr = 6*np.abs(np.random.randn(n,3))
  dim_arr = np.full((n,3), 2)

  st = time()

  hdg = obstacle_avoidance(centr_arr, dim_arr, 43)
  print(hdg)
  
  print(time()-st)