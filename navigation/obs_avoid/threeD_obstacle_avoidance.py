# obstacle avoidance, updated for 3d and vectorized
import numpy as np
from time import time


def sweep_quadrant(max_hdg, increment, mag_vec, radius_vec, Xs, Ys, path_width):
  # step is + when sweeping right, - when sweeping left
  step = increment * max_hdg / np.abs(max_hdg)
  hdg_range = np.arange(0, max_hdg, step)
  hdg_arr = np.tile(hdg_range, (len(mag_vec),1)).T

  # calculate the path inequalities for each heading angle for each obstacle
  phi_arr = np.abs(hdg_arr - (90 - np.degrees(np.arctan(Ys/Xs))))
  D_diff_arr = mag_vec*(np.sin(np.radians(phi_arr))) - radius_vec
  
  # mask any dangerous obstacles and their whole row (which corresponds to a heading angle)
  masked_D_diff_arr = np.ma.masked_less_equal(D_diff_arr, path_width)
  masked_rows = np.ma.mask_rows(masked_D_diff_arr)
  bool_arr = masked_rows.mask

  # the first index with False corresponds to the smallest safe heading
  try: hdg_output = np.where(bool_arr == False)[0][0]
  except IndexError: return max_hdg
  return step*hdg_output
  


def check_Zs(centrs, dims, path_height):
  # Masks to check the positivity or negativity 
  pos_mask = np.logical_not(
    np.zeros(centrs.shape, bool) | (centrs[:,2] < 0)[:, None]
  )
  neg_mask = np.logical_not(
    np.zeros(centrs.shape, bool) | (centrs[:,2] >= 0)[:, None]
  )

  # Masks to check the danger with respect to path height
  # The inequality condition is different for pos or neg Zs
  pos_ineq_mask = np.logical_not(
    np.zeros(centrs.shape, bool) | \
    (centrs[:,2]-dims[:,2] > path_height)[:, None]
  )
  neg_ineq_mask = np.logical_not(
    np.zeros(centrs.shape, bool) | \
    (centrs[:,2]+dims[:,2] < -path_height)[:, None]
  )

  # Intersect pos/neg masks with their respective danger masks
  # Union the remaining 'Trues' as they are the relevant centroids
  # Reverse the result since numpy masked arrays read 'False' as unmasked values
  combined_pos_mask = pos_mask & pos_ineq_mask
  combined_neg_mask = neg_mask & neg_ineq_mask
  overall_mask = np.logical_not(
    combined_pos_mask | combined_neg_mask
  )

  # Mask the original arrays 
  masked_centrs = np.ma.array(centrs, mask=overall_mask)
  masked_dims = np.ma.array(dims, mask=overall_mask)

  # Extract non-masked data while maintaining array structure
  new_centrs = np.ma.getdata(
    masked_centrs[~masked_centrs.mask.any(axis=1)]
  )
  new_dims = np.ma.getdata(
    masked_dims[~masked_dims.mask.any(axis=1)]
  )
  return new_centrs, new_dims


def obstacle_avoidance(centr_arr, dim_arr, max_hdg):
  increment = 0.1   # deg
  path_width = 1  # m
  safe_dist = 1   # m
  path_height = 1 # m

  # Check which objects are within a dangerous height
  new_centrs, new_dims = check_Zs(centr_arr, dim_arr, path_height)

  # approximate max len dimension as sphereical radius if no dangers, skip obstacle avoidance
  try:
    radius_vec = np.amax(new_dims, axis=1) 
  except np.AxisError:
    return False
  
  # get separate x and y coords and magnitudes of their coordinates
  twoD_centrs = np.delete(new_centrs, 2, axis=1)
  x_centrs = twoD_centrs[:,0]
  y_centrs = twoD_centrs[:,1]
  mag_vec = np.sqrt(np.sum(np.abs(twoD_centrs)**2,axis=1))

  # sweep right and left quadrants
  right_hdg = sweep_quadrant(
    max_hdg, increment, mag_vec, radius_vec, x_centrs, y_centrs, path_width
  )
  left_hdg = sweep_quadrant(
    -max_hdg, increment, mag_vec, radius_vec, x_centrs, y_centrs, path_width
  )

  # return the smallest magnitude turn
  if abs(left_hdg) < right_hdg: return left_hdg
  else: return right_hdg


if __name__=='__main__':
  #centr_arr = np.array([[2, 16, 1],
  #                      [-5, 7, -1]] )
  #dim_arr = np.full((2,3), 1)
  n=10
  centr_arr = 8*np.abs(np.random.randn(n,3))
  dim_arr = np.full((n,3), 2)
  
  print(centr_arr)
  st = time()

  hdg = obstacle_avoidance(centr_arr, dim_arr, 43)
  print(hdg)
  
  print(time()-st)