import numpy as np

class DroneState:
    def __init__(self, x, y, z, vx, vy, vz, ax, ay, az):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.ax = ax
        self.ay = ay
        self.az = az

class KalmanFilter:
    def __init__(self, init_state: DroneState):
        self._iter = 0
        self._cur_state = init_state # initial system state
        self._p_state = DroneState(0, 0, 0, 0, 0, 0, 0, 0, 0) # initial state variance

    def state_update(self, z: DroneState, r: DroneState):
        pass

    def kalman_gain(self):
        pass

class KalmanExample:
    def __init__(self, init_height: float, init_std_dev: float, r: float, q: float):
        self._iter = 1
        self._height = init_height
        self._p = init_std_dev ** 2
        self._r = r ** 2
        self._q = q
    
    def update(self, z: float):
        next_height = self._height
        next_p = self._p + self._q
        kalman_gain = next_p / (next_p + self._r)
        self._height = next_height + kalman_gain * (z - next_height)
        self._p = (1 - kalman_gain) * next_p
        self._iter += 1

    def cur_state(self):
        return self._height
    
    def p(self):
        return self._p
    
# Test Kalman Paper Example
kf = KalmanExample(60, 100, 0.1, 0.0001)
data = [49.986, 49.963, 50.09, 50.001, 50.018, 50.05, 49.938, 49.858, 49.965, 50.114]

for z in data:
    kf.update(z)
    print(kf.cur_state())
    print(kf.p())
    print()
    





    
