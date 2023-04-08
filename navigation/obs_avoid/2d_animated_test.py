# drone animation test
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from obs_avoid_animation import *
#from test2 import *

LOWER_LIMIT =-30
UPPER_LIMIT=30

class LivePlot:


    def __init__(self, num_obj):    # num_markers should be number of objects 
        self.x_lim = np.array([LOWER_LIMIT, UPPER_LIMIT]) # dimensions of plot, ex: x-coords from 0 to 100
        self.y_lim = np.array([LOWER_LIMIT, UPPER_LIMIT]) # make them similar to range of d455 camera, the y-lim is 20 meters

        self.obj_marker = 'ro'          # matplotplib parameter for marker color and shape
        self.drone_marker = 'b^' 

        self.x_coords = np.zeros(num_obj)  # correspond each index in these arrays to an object
        self.y_coords = np.zeros(num_obj)  # the final index represents the drone itself
        self.x_drone = [0]
        self.y_drone = [0]

        self.init_plot()
        

    def init_plot(self):                
        self.fig, self.ax1 = plt.subplots(figsize=(8, 7))   # initialize matplotlib figure and window size
        self.ax1.set_xlim(self.x_lim[0], self.x_lim[1])     # initialize figure dimensions
        self.ax1.set_ylim(self.y_lim[0], self.y_lim[1])

        plt.show(block = False)                             # stuff for animations
        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.fig.canvas.blit(self.fig.bbox)


    def update(self, position, id=0, is_drone=False):    # is_drone=True to plot the drone, false for objects
        if is_drone == False:                            # id corresponds to an index in array of object coords
            self.x_coords[id] = position[0]                 
            self.y_coords[id] = position[1]
            marker = self.obj_marker
        else:
            self.x_coords[len(self.x_coords)-1] = position[0]
            self.y_coords[len(self.y_coords)-1] = position[1]
            marker = self.drone_marker
                                                    
        self.fig.canvas.restore_region(self.bg)
        for i in range(len(self.x_coords) - 1):         # iterate thru coord array and draw each one
            (pt,) = self.ax1.plot(
                self.x_coords[i], self.y_coords[i], 
                self.obj_marker, markersize=5)
            pt.set_data(self.x_coords, self.y_coords)

            (pt2,) = self.ax1.plot(
                self.x_drone[0], self.y_drone[0], 
                self.drone_marker, markersize=5)
            self.ax1.set_xlim(self.x_lim[0], self.x_lim[1])
            self.ax1.set_ylim(self.y_lim[0], self.y_lim[1])
            #self.ax1.draw_artist(pt)    
            #self.ax1.draw_artist(pt2)    
           # print("===sleeping for a moment===")
         #   time.sleep(0.2)


        # animation stuff
        #self.fig.canvas.blit(self.fig.bbox)
        #self.fig.canvas.flush_events()


    def update_drone(self, position):    # is_drone=True to plot the drone, false for objects
        
        self.x_drone[0] = position[0]
        self.y_drone[0] = position[1]
        marker = self.drone_marker
                                                    
        self.fig.canvas.restore_region(self.bg)
        (pt,) = self.ax1.plot(
                self.x_coords, self.y_coords, 
                self.obj_marker, markersize=5)



        (pt2,) = self.ax1.plot(
                self.x_drone[0], self.y_drone[0], 
                self.drone_marker, markersize=5)
        self.ax1.set_xlim(self.x_lim[0], self.x_lim[1])
        self.ax1.set_ylim(self.y_lim[0], self.y_lim[1])
        self.ax1.draw_artist(pt)
        self.ax1.draw_artist(pt2)    
        # animation stuff
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()


if __name__ == '__main__':
    num = 3
    new_plot = LivePlot(num)
    TIMESTEP = 0.5
    ANIMATION_LENGTH = 50
    print(f'timestep: {TIMESTEP}')
    centroids, relative_positions, dimensions, velocities = create_obstacles()
    drone_heading = 0 # straight forward (up).
    drone_position = [0,0] # start at (0,0)
    t=time.time()
    while (time.time() - t ) < ANIMATION_LENGTH:

        # find best course:
        if obstacles_within_FOV(obstacle_positions=relative_positions):
            heading = obstacle_avoidance(relative_positions, dimensions)
            print('avoiding obstacles')
        else:
            heading = head_towards_goal(drone_position=drone_position, drone_heading=drone_heading)
            print('no obstacles seen in fov, heading towards goal')
        
        drone_heading = update_drone_heading(drone_heading=drone_heading, heading=heading)
        print(f'drone heading: {drone_heading}, changed by {heading} degrees')
        if (drone_heading > 360): 
            drone_heading = drone_heading - 360
            print('for clarity, changed drone angle to {drone_heading}')
        elif (drone_heading < -360): 
            drone_heading = drone_heading + 360
            print('for clarity, changed drone angle to {drone_heading}')

        # update positions:
        update_drone_position(drone_heading, drone_position)
        print(f'drone position updated: {drone_position}')
        update_obstacle_positions(centroids, velocities)
        print(f'obstacles at: {centroids}')
        update_relative_positions(relative_positions, centroids, drone_position, drone_heading)
        print(f'relative obstacle positions: {relative_positions}')
        print('loop complete')
        print()
        
        for i in range(num):
            new_plot.update(centroids[i], id=i)
    
        new_plot.update_drone(drone_position)
        

        time.sleep(TIMESTEP) 
         
        
