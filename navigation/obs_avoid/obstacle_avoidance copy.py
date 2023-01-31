import math
import numpy as np


def distance(pos):
  xy_pos = np.array([pos[0], pos[1]])
  dist = np.sqrt(xy_pos.dot(xy_pos))
  return dist


def avoidObstacle(centroids, box_dims): #arrays of object centroids and bounding box dimensions
  R = 10 #meters, radius of field view
  angle = 45 #degrees, field view angle
  path_distance = 3 #meters, safety distance from the path
  drone_distance = 3 #meters, safety distance from drone

  cur_heading = 0 #degrees, drone's current heading
  avoidObstacle = False

  for i in np.size(centroids, axis=0):
    drone_dist = distance(centroids[i]) #use radial distance function 
    path_dist = abs(centroids[i][0]) 

    if (path_dist < np.amax(box_dims[i]) + path_distance) or \
       (drone_dist < drone_distance + np.amax(box_dims[i])):  
      avoidObstacle = True
      break
  
  while avoidObstacle and cur_heading < angle:
    cur_heading += 10
    avoidObstacle = False

    for i in np.size(centroids, axis=0):
      drone_dist = distance(centroids[i]) #use  radial distance function 
      beta = math.atan(centroids[i][0]/centroids[i][1]) - math.radians(cur_heading)
      path_dist = drone_dist * math.sin(beta) #take absolute value here

      if (path_dist < np.amax(box_dims[i]) + path_distance) or \
         (drone_dist < drone_distance + np.amax(box_dims[i])): #safety distance from path 
        avoidObstacle = True
        break

  if(cur_heading > 45):
    return 45
  else:
    return cur_heading


def main(centers, dimensions):
  objects = []
  for c, d in zip(centers, dimensions):
    obs = obstacle(c, d)
    objects.append(obs)

  drone_heading = avoidObstacle(objects)
  print(drone_heading)

if __name__ == " __main__":
    centers = [[0,5]]
    dimensions = [0.25]
    main(centers, dimensions)