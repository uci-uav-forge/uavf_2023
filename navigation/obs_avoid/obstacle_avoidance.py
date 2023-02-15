import math
import numpy as np

def obstacle_avoidance(centeroids, dimensions):
  #R = 10
  angle = 45
  path_distance = 1
  drone_distance = 1

  cur_heading = 0
  obstacle_avoidance = False

  radius_list = np.amax(dimensions, axis=1) #list of all the radius's
  distance_list = np.linalg.norm(centeroids, axis=1) #distance of all the objects
  path_dist = np.abs(centeroids[:,0]) #list of all the abs(x) values
  if(np.any(path_dist< radius_list+path_distance) or np.any(distance_list < radius_list + drone_distance)):
    obstacle_avoidance = True

  while obstacle_avoidance and cur_heading < angle:
    cur_heading += 10
    obstacle_avoidance = False


    radius_list = np.amax(dimensions, axis=1) #list of all the radius's
    distance_list = np.linalg.norm(centeroids, axis=1) #distance of all the objects
    beta_list = np.arctan2(centeroids[:,0], centeroids[:,1]) - np.radians(cur_heading)
    path_dist = np.abs(distance_list * np.sin(beta_list))
    if(np.any(path_dist< radius_list+path_distance) or np.any(distance_list < radius_list + drone_distance)):
      obstacle_avoidance = True
    
  if(cur_heading > 45):
    return 45
  else:
    return cur_heading



def main(centers, dimensions):
  drone_heading = obstacle_avoidance(centers, dimensions)
  print(drone_heading)


if __name__ == " __main__":
  centers = np.array([[0,6], [0,5]])
  dimensions = np.array([[0.25, 0.1, 0.1], [1.0, 0.25, 0.25]])
  main(centers, dimensions)