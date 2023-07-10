import numpy as np
import rospy
from std_msgs.msg import Int16MultiArray
import time
from collections import deque, defaultdict

HIST_THRESH = 2
HIST_TTL = 2 # seconds.
TIMING = True

class ObstacleHistogram:
    def __init__(self):
        self.hist = defaultdict(deque)
        self.pcd_sub = rospy.Subscriber(
            name="obs_avoid_pcd",
            data_class=Int16MultiArray,
            callback = self.recv_points
        )
        self.rel_pts = set()
    
    def recv_points(self, pts):
        if TIMING:
            t0 = time.time()

        pts_np = list(map(tuple,(np.array_split(np.array(pts.data), len(pts.data)//3))))
        self.enter_pts(pts_np)
        
        if TIMING:
            t1 = time.time()
            print(t1 - t0)

            t0 = time.time()
            pts = self.get_rel_pts()
            print(list(pts)[:5])
            t1 = time.time()
            print(t1-t0, len(pts))
    
    def enter_pts(self, pts):
        for p in pts:
            self.hist[p].append(time.time())
            if len(self.hist[p]) >= HIST_THRESH:
                self.rel_pts.add(p)
    
    def clean_hist(self, p):
        while len(self.hist[p]) and HIST_TTL < time.time() - self.hist[p][0]:
                self.hist[p].popleft()
        if len(self.hist[p]) < HIST_THRESH:
            rms.append(p)

    def get_rel_pts(self): 
        rms = []
        for p in self.rel_pts:
            self.clean_hist(p)
        for p in rms:
            self.rel_pts.remove(p)
        return self.rel_pts

    def get_confidence(self, pos):
        self.clean_hist(pos)
        return len(self.hist[pos])/HIST_THRESH