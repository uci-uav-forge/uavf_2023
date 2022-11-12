from queue import PriorityQueue
import numpy as np
import math
import rospy
from iq_gnc.py_gnc_functions import *
from PrintColours import *


# mission parameters
Z0 = 50
Zdrop = 5

vavg = 10
vdrop = 2

pq = PriorityQueue()

# init drone
rospy.init_node("drone_controller", anonymous=True)
drone = gnc_api()
drone.wait4connect()
drone.wait4start()

# load waypoints into queue
goals = [[0, 0, 3, 0], [5, 0, 3, -90], [5, 5, 3, 0],
             [0, 5, 3, 90], [0, 0, 3, 180], [0, 0, 3, 0]]

for goal in goals:
    goal_point = Point(goal, 'WAYPOINT')
    dist = goal_point.distance(ORIGIN)
    pq.put((10**9 - dist, goal_point))

# drone takeoff
drone.initialize_local_frame()
drone.takeoff(z0)
rate = rospy.Rate(3)


# outer loop: check if there are more waypoints to travel to
while not pq.empty():
    cur_waypoint = pq.queue[0]
    cw_x, cw_y, cw_z, cw_ang = cur_waypoint.coords()

    drone.set_destination(
        x=cw_x, y=cw_y, z=cw_z, psi=cw_ang)

    # get next waypoint, if obs/drop detected, go there
    while not drone.check_waypoint_reached():
        next = pq.queue[0]
        if next != cur_waypoint:
            cur_waypoint = next
            n_x, n_y, n_z, n_ang = next.coords()
            drone.set_destination(
                x=n_x, y=n_y, z=n_z, psi=n_ang)
    
    

