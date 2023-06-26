import numpy as np
import rospy
from std_msgs.msg import Int16MultiArray
import time
from collections import deque

HIST_THRESH = 2
HIST_TTL = 2 # seconds.
TIMING = True

class ObstacleHistogram:
    def __init__(self, width, height, depth):
        self.width = 2*width
        self.height = 2*height
        self.depth = 2*depth
        self.hist = np.zeros((self.width, self.height, self.depth), dtype = np.byte)
        self.pcd_sub = rospy.Subscriber(
            name="obs_avoid_pcd",
            data_class=Int16MultiArray,
            callback = self.recv_points
        )
        self.rel_pts = set()
        self.rm_queue = deque()
    
    def recv_points(self, pts):
        if TIMING:
            t0 = time.time()
        
        pts_np = np.array(np.array_split(np.array(pts.data), len(pts.data)//3))
        pts_np += np.array([self.width//2, self.height//2, self.depth//2])
        self.enter_pts(pts_np)
        self.rm_queue.append((time.time(), pts_np))
        while time.time() - self.rm_queue[0][0] > HIST_TTL:
            self.rm_pts(self.rm_queue.popleft()[1])
        
        if TIMING:
            t1 = time.time()
            print(t1 - t0)

            t0 = time.time()
            pts = self.get_rel_pts()
            print(list(pts)[:5])
            t1 = time.time()
            print(t1-t0, len(pts))
    
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
        return np.array(list(self.rel_pts)) - np.array([self.width//2, self.height//2, self.depth//2])

if __name__ == '__main__':
    rospy.init_node('obs_hist_test')
    h = ObstacleHistogram(500,50,500)
    rospy.spin()