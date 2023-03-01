import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import time


class LivePlot:
    

    def __init__(self, num_markers):    # num_markers should be number of objects 
        self.x_lim = np.array([0, 100]) # dimensions of plot, ex: x-coords from 0 to 100
        self.y_lim = np.array([0, 100]) # make them similar to range of d455 camera, the y-lim is 20 meters

        self.obj_marker = 'ro'          # matplotplib parameter for marker color and shape
        self.drone_marker = 'b^' 

        self.x_coords = np.zeros(num_markers+1)  # correspond each index in these arrays to an object
        self.y_coords = np.zeros(num_markers+1)  # the final index represents the drone itself

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
                marker, markersize=5)
            pt.set_data(self.x_coords[i], self.y_coords[i])

            self.ax1.set_xlim(self.x_lim[0], self.x_lim[1])
            self.ax1.set_ylim(self.y_lim[0], self.y_lim[1])
            self.ax1.draw_artist(pt)    

        # animation stuff
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()


    def close(self):
        plt.close()


if __name__ == '__main__':
    new_plot = LivePlot(2)
    while True:
        for i in range(3):
            new_plot.update(np.array(
                [random.randrange(0, 2, 1),
                 random.randrange(0, 2, 1),
                random.randrange(0, 2, 1)]), 
                id=i)
            
            new_plot.update(np.array(
                [random.randrange(0, 2, 1),
                 random.randrange(0, 2, 1),
                random.randrange(0, 2, 1)]),
                is_drone=True
            )