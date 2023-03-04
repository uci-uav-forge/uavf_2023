# test obstacle avoidance in matlplotlib, animated
# requires obstacle_avoidance.py and live_plot.py to be in the same folder
# Note: (0,0) is where the drone starts flying. Forward is moving up (y-direction), based on how live_plot seems to be structured.

from live_plot import *
from obstacle_avoidance import *
import numpy as np
import time
import math

TIMESTEP = 0.5 # seconds
MAX_TURN = 180 # the maximum a drone can turn in one second
DRONE_SPEED = 1 # all speeds in m/s

# main code:
def run_test():
    centroids, dimensions, velocities = create_obstacles() # change this function to change obstacles
    drone_heading = 0 # straight forward (up). 90 degrees would be right, -90 (or 270) left. 
    drone_position = [0,0] # start at (0,0)
    relative_positions = [x[:] for x in centroids] # complicated way of making copy so same list is not referenced.
    # centroids will keep track of obstacles from a global frame of a reference. we also need relative_positions
    # to keep track of obstacles from the drone's point of view, as obstacle_avoidance is a relative function

    t = time.time()
    while time.time() - t < 10: # kill animation after 60 seconds

        # find best course:
        heading = obstacle_avoidance(centroids, dimensions)
        drone_heading = update_drone_heading(drone_heading=drone_heading, heading=heading)

        # update positions:
        update_drone_position(drone_heading, drone_position)
        update_obstacle_positions(centroids, velocities)
        update_relative_positions(relative_positions, centroids, drone_position) # ***this function is not done yet***


        time.sleep(TIMESTEP)

    
def create_obstacles() -> None:
    centroids = np.array([[1,1], [2,1], [-4,-5]])       # location on plot (x,y)
    dimensions = np.array([(1,1,1), (2,1,2), (4,2,3)])  # size (x,y,z)
    velocities = [(0.5, 0.5), (1,1), (-1, 2)]           # speed(vx, vy).
    return centroids, dimensions, velocities

def update_drone_heading(drone_heading: int, heading: int):
    '''update drone heading, taking into account that drone has max turn rate'''
    max_turn = MAX_TURN*TIMESTEP
    if abs(heading - drone_heading) < max_turn:
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
        centroids[i][0] = centroids[i][0] + velocities[i][0]*TIMESTEP
        centroids[i][1] = centroids[i][1] + velocities[i][1]*TIMESTEP

def update_relative_positions(relative_position: list, centroids: list, drone_position: int):
    pass
    

if __name__ == '__main__':
    run_test()