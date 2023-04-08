# test obstacle avoidance in animated
# requires obstacle_avoidance.py and live_plot.py to be in the same folder
# see bottom for important notes on setup
# Note: live_plot.py has the actual animation, but parameters and functions are all here.
from live_plot import *
from obstacle_avoidance import *
import numpy as np
import time
import math

TIMESTEP = 1 # seconds
MAX_TURN = 180 # the maximum a drone can turn in one second (degrees)
DRONE_SPEED = 1 # m/s
ANIMATION_LENGTH = 5 # How long the animation should last, in seconds.
DESTINATION  = (0,30)

# main code:
def run_test():
    centroids, relative_positions, dimensions, velocities = create_obstacles() # change this function to change obstacles
    # Note: centroids variable will keep track of where obstacles are from a global frame of a reference. we also need relative_positions
    # to keep track of obstacles from the drone's point of view, as obstacle_avoidance() assumes the drone's perspective.
    drone_heading = 0 # straight forward (up).
    drone_position = [0,0] # start at (0,0)

    t = time.time()
    while time.time() - t < ANIMATION_LENGTH:

        # find best course:
        if obstacles_within_FOV(obstacle_positions=relative_positions):
            heading = obstacle_avoidance(relative_positions, dimensions)
            print('avoiding obstacles')
        else:
            heading = head_towards_goal(drone_position=drone_position, drone_heading=drone_heading)
            print('no obstacles seen in fov, heading towards goal')
        
        drone_heading = update_drone_heading(drone_heading=drone_heading, heading=heading)
        print(f'drone heading: {drone_heading}, changed by {heading} degrees')

        # update positions:
        update_drone_position(drone_heading, drone_position)
        print(f'drone position updated: {drone_position}')
        update_obstacle_positions(centroids, velocities)
        print(f'obstacles at: {centroids}')
        update_relative_positions(relative_positions, centroids, drone_position, drone_heading)
        print(f'relative obstacle positions: {relative_positions}')
        print('loop complete')
        print()

        time.sleep(TIMESTEP)

    
def create_obstacles() -> None:
    centroids = np.array([[1,0.5], [2,1], [-4,-5]], dtype=float)       # location on plot (x,y)
    relative_positions = np.array([[2,2], [3,1], [-4,-5]]) # *** must match centroids***, since drone begins at (0,0) they match.
    dimensions = np.array([(1,1,1), (2,1,2), (4,2,3)])  # size (x,y,z)
    velocities = [(-1, 1), (2,1), (-1, 2)]           # speed(vx, vy).
    return centroids, relative_positions, dimensions, velocities

def update_drone_heading(drone_heading: int, heading: int):
    '''update drone heading, taking into account that drone has max turn rate'''
    max_turn = MAX_TURN*TIMESTEP
    if abs(heading) < max_turn: # drone capable of turning as much as function tells it to
        drone_heading = drone_heading + heading
    elif (heading > 0):
        drone_heading = drone_heading + max_turn
    else:
        drone_heading = drone_heading - max_turn
    
    return drone_heading

def update_drone_position(heading: int, position: list):
    '''based on provided heading, will drones will update position'''
    position[0] = position[0] + DRONE_SPEED*math.sin(math.radians(heading))*TIMESTEP
    position[1] = position[1] + DRONE_SPEED*math.cos(math.radians(heading))*TIMESTEP

def update_obstacle_positions(centroids: list, velocities: list):
    '''based on provided velocity in (vx,vy) format, will update position'''
    for i in range(len(centroids)):
        #print(f'obstacle at {centroids[i]} will go forward by {velocities[i]}')
        centroids[i][0] = centroids[i][0] + velocities[i][0]*TIMESTEP
        centroids[i][1] = centroids[i][1] + velocities[i][1]*TIMESTEP
        #print(f'obstacle has now moved to {centroids[i]}')

def update_relative_positions(relative_position: list, centroids: list, drone_position: int, drone_heading: int):
    '''update where obstacles are, from viewpoint of drone'''
    for i in range(len(centroids)):
        a1 = math.atan((centroids[i][0]-drone_position[0]) / (centroids[i][1]-drone_position[1])) # calculate angle of line from drone to obstacle. angle = arctan(x/y)
        a2 = a1 - math.radians(drone_heading) # angle difference between heading of drone and direction to obstacle
        dist = math.dist(centroids[i], drone_position) # distance between drone and obstacle
        #dist = math.sqrt( (centroids[i][1]-drone_position[1])**2 + (centroids[i][0]-drone_position[0])**2 ) # distance between drone and obstacle
        relative_position[i][0] = dist*math.sin(a2) # x value
        relative_position[i][1] = dist*math.cos(a2) # y value

        # temporary, for debugging:
        '''print(f'debug: drone heading is {drone_heading} degrees and position is {drone_position}')
        print(f'debug: angle of object from drone: {math.degrees(a1)} ')
        print(f'debug: angle used to calculate relative position: {math.degrees(a2)}')
        print(f'debug: distance to object: {dist}')
        print(f'debug: calculated parameters: x = {relative_position[i][0]}, y = {relative_position[i][1]} ')
        print()'''

def obstacles_within_FOV(obstacle_positions: np.array) -> bool:
    '''should obstacle avoidance even be run? check if obstaces are within view'''
    for centroid in obstacle_positions:
        #print(f'FOV debug: looking at obstacle with relative position {centroid}')
        if centroid[0] == 0 and centroid[1] == 0: return True # so won't crash if drone somehow is on top of obstacle 

        #if centroid[1] == 0: a = 0
        a = math.atan2(centroid[0], centroid[1]) # angle from drone to obstacle (from y axis) is arctan(x/y)
        if abs(a) < 0.785: # angle less than 45 degrees
            d = math.dist(centroid, (0,0))
            if d < 16:
                #print(f'FOV debug: obstacle {centroid} found within fov')
                return True
    
    return False

def head_towards_goal(drone_position: list, drone_heading: int) -> int:
    '''tells drone where to turn if it wants to get to goal'''
    #print(f'goal debug: drone heading is {drone_heading} and position is {drone_position}')
    max_turn = MAX_TURN*TIMESTEP
    angle = math.atan2((DESTINATION[0] - drone_position[0]),(DESTINATION[1] - drone_position[1])) # angle of vector from drone to goal
    angle = math.degrees(angle)
    #print(f'goal debug: calculated angle: {angle}')
    angle = angle - drone_heading
    
    #print(f'telling drone to go {angle}')
    return angle


if __name__ == '__main__':
    run_test()

'''
Note: (0,0) is where the drone starts flying
Note 2: angle is phi that diverges from the y axis. so heading = 10 means turn 10 degrees to the right. ALL ANGLES are from y axis.
Note 3: coordinates are stored (x,y). Velocities are stored (vx, vy). dimensions are stored (x,y,z)
'''