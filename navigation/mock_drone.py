import time
from threading import Thread
import numpy as np
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float

class MockDrone:
    def __init__(self):
        self.location = [0, 0, 0]
        self.angles = [0, 0, 0]
        self.destination = [0, 0, 0]
    def wait4connect(self):
        time.sleep(0.2)
    def set_mode_px4(self, mode):
        time.sleep(0.2)
        print(f"set mode to {mode}")
    def set_speed_px4(self, speed):
        time.sleep(0.2)
        print(f"set speed to {speed}")
    def initialize_local_frame(self):
        time.sleep(0.2)
    def get_current_location(self):
        return Point(*self.location)
    def get_current_xyz(self):
        return self.location
    def arm(self):
        time.sleep(0.5)
    def set_destination(self, x, y, z, psi):
        self.destination = [x, y, z]
        def goto_destination():
            time.sleep(3)
            self.location = [x, y, z]
            self.angles = [0, 0, psi]
        Thread(target=goto_destination).start()
    def check_waypoint_reached(self):
        return np.linalg.norm(np.array(self.location) - np.array(self.destination)) < 0.1
    def land(self):
        time.sleep(1)
        print("Landed")
    def get_current_pitch_roll_yaw(self):
        return self.angles
    def get_current_pos_and_angles(self):
        return self.location, self.angles

    