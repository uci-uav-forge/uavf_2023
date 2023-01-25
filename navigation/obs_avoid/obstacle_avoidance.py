import math
class obstacle:
  def __init__(self, pos, rad):
    self.position = pos
    self.radius = rad
    self.velocity = [0, 0, 0]
  
  def get_position(self):
    return self.position
  
  def set_position(self, pos):
    self.position = pos
  
  def get_radius(self):
    return self.radius
  
  def set_radius(self, rad):
    self.radius = rad

  def get_velocity(self):
    return self.velocity
  
  def set_velocity(self, vel):
    self.velocity = vel

def distance(position):
  x,y = position
  dist = math.sqrt(x**2 + y**2)
  return dist



def avoidObstacle(objects): #objects should be a list of obstacles
  R = 10 #meters, radius of field view
  angle = 45 #degrees, field view angle
  path_distance = 3 #meters, safety distance from the path
  drone_distance = 3 #meters, safety distance from drone

  cur_heading = 0 #degrees, drone's current heading
  avoidObstacle = False

  for obj in objects:
    drone_dist = distance(obj.get_position()) #use radial distance function 
    path_dist = abs(obj.get_position()[0]) 
    if (path_dist < obj.get_radius() + path_distance) or (distance(obj.get_position()) < drone_distance + obj.get_radius()):  
      avoidObstacle = True
      break

  
  while avoidObstacle and cur_heading < angle:
    cur_heading += 10
    avoidObstacle = False

    for obj in objects:
      drone_dist = distance(obj.get_position()) #use  radial distance function 
      beta = math.atan(obj.get_position()[0]/obj.get_position()[1]) - math.radians(cur_heading)
      path_dist = drone_dist * math.sin(beta) #take absolute value here
      if (path_dist < obj.get_radius() + path_distance) or (drone_dist < drone_distance + obj.get_radius()): #safety distance from path 
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