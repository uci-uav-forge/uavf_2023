# test obstacle avoidance in matlplotlib, animated
# requires obstacle_avoidance.py and live_plot.py to be in the same folder
# Note: (0,0) is where the drone starts flying. Forward is moving up (y-direction), based on how live_plot seems to be structured.

from live_plot import *
from obstacle_avoidance import *
import numpy as np
import time
import math

TIMESTEP = 1 # seconds
MAX_TURN = 180 # the maximum a drone can turn in one second (degrees)
DRONE_SPEED = 1 # m/s
ANIMATION_LENGTH = 5 # How long the animation should last, in seconds.

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
        heading = obstacle_avoidance(centroids, dimensions)
        drone_heading = update_drone_heading(drone_heading=drone_heading, heading=heading)
        #print(f'drone heading: {drone_heading}, drone position: {drone_position}')
        #print()

        # update positions:
        update_drone_position(drone_heading, drone_position)
        update_obstacle_positions(centroids, velocities)
        #print(f'obstacles just got updated: {centroids}')
        update_relative_positions(relative_positions, centroids, drone_position, drone_heading)


        time.sleep(TIMESTEP)

    
def create_obstacles() -> None:
    centroids = np.array([[1,1], [2,1], [-4,-5]], dtype=float)       # location on plot (x,y)
    relative_positions = [[1,1], [2,1], [-4,-5]] # *** must match centroids***, since drone begins at (0,0) they match.
    dimensions = np.array([(1,1,1), (2,1,2), (4,2,3)])  # size (x,y,z)
    velocities = [(0.5, 0.5), (1,1), (-1, 2)]           # speed(vx, vy).
    return centroids, relative_positions, dimensions, velocities

def update_drone_heading(drone_heading: int, heading: int):
    '''update drone heading, taking into account that drone has max turn rate'''
    max_turn = MAX_TURN*TIMESTEP
    if abs(heading - drone_heading) < max_turn: # drone capable of turning as much as function tells it to
        drone_heading = heading
    elif (heading - drone_heading > 0):
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
        a1 = math.atan((centroids[i][1]-drone_position[1]) / (centroids[i][0]-drone_position[0])) # calculate angle of line from drone to obstacle
        a2 = math.radians(drone_heading) - a1 # angle difference between heading of drone and direction to obstacle
        dist = math.sqrt( (centroids[i][1]-drone_position[1])**2 + (centroids[i][0]-drone_position[0])**2 ) # distance between drone and obstacle
        relative_position[i][0] = dist*math.cos(a2) # x value
        relative_position[i][1] = dist*math.sin(a2) # y value

        # temporary, for debugging:
        #print(f'distance to object: {dist}')
        #print(f'calculated parameters: x = {dist*math.cos(a2)}, y = {dist*math.sin(a2)} ')
        
        
    
    

if __name__ == '__main__':
    run_test()