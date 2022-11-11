from queue import PriorityQueue
import math
from iq_gnc.py_gnc_functions import *
import rospy

class Point:
    def __init__(self, coords: tuple, type: str):
        self.x, self.y, self.z, self.ang = coords
        self.type = type
    
    def distance(self, other: 'Point'):
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def coords(self):
        return (self.x, self.y, self.z, self.ang)

    

ORIGIN = Point(coords=(0, 0, 0, 0), type='WAYPOINT')

z0 = 50
zdrop = 5

vavg = 3
vdrop = 2

pq = PriorityQueue()

rospy.init_node("drone_controller", anonymous=True)
drone = gnc_api()
drone.wait4connect()
drone.wait4start()

goals = [[0, 0, 3, 0], [5, 0, 3, -90], [5, 5, 3, 0],
             [0, 5, 3, 90], [0, 0, 3, 180], [0, 0, 3, 0]]

for goal in goals:
    goal_point = Point(goal, 'WAYPOINT')
    dist = goal_point.distance(ORIGIN)
    pq.put((10**9 - dist, goal_point))

drone.initialize_local_frame()
drone.takeoff(z0)
rate = rospy.Rate(3)



while not pq.empty():
    pass
