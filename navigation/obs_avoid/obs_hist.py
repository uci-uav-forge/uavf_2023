import numpy as np
import rospy
from std_msgs.msg import Int16MultiArray
import time
from collections import deque

HIST_WIDTH = 200 # meters.
HIST_THRESH = 2
HIST_TTL = 2 # seconds.

class ObstacleHistogram:
    def __init__(self):
        self.hist = np.zeros((HIST_WIDTH, HIST_WIDTH, HIST_WIDTH))
        self.pcd_sub = rospy.Subscriber(
            name="obs_avoid_pcd",
            data_class=Int16MultiArray,
            callback = self.recv_points
        )
        self.rel_pts = set()
        self.rm_queue = deque()
    
    def recv_points(self, pts):
        pts_np = np.array(np.array_split(np.array(pts.data), len(pts.data)//3))
        pts_np += HIST_WIDTH//2
        self.enter_pts(pts_np)
        self.rm_queue.append((time.time(), pts_np))
        while time.time() - self.rm_queue[0][0] > HIST_TTL:
            self.rm_pts(self.rm_queue.popleft()[1])
        #print(self.get_rel_pts())
    
    def enter_pts(self, pts):
        np.add.at(self.hist, (pts[:,0], pts[:,1], pts[:,2] ), 1)
        for i, v in enumerate(self.hist[pts[:,0], pts[:,1], pts[:,2] ]):
            if v >= HIST_THRESH:
                self.rel_pts.add(tuple(pts[i]))
    
    def rm_pts(self, pts):
        np.add.at(self.hist, (pts[:,0], pts[:,1], pts[:,2] ), -1)
        for i, v in enumerate(self.hist[pts[:,0], pts[:,1], pts[:,2] ]):
            if v < HIST_THRESH:
                self.rel_pts.discard(tuple(pts[i]))

    def get_rel_pts(self):
        return np.array(list(self.rel_pts)) - HIST_WIDTH//2

if __name__ == '__main__':
    rospy.init_node('obs_hist_test')
    h = ObstacleHistogram()
    rospy.spin()