from math import atan2, asin, degrees

import rospy
from scipy.spatial.transform import Rotation
import numpy as np
from tf import transformations
from navigation.guided_mission.py_gnc_functions import gnc_api

rospy.init_node("drone_GNC", anonymous=True)
drone = gnc_api()
#drone.initialize_local_frame()

while 1:
    angles = drone.get_current_pitch_roll_yaw()
    print(angles)
    q0, q1, q2, q3 = drone.get_orientation_quaternion_xyzw() 
    if np.linalg.norm([q0,q1,q2,q3])==0:
        print("zero norm quaternion")
        continue
    print(f"quaternion:\t{[q0,q1,q2,q3]}")
    print(f"camera angles:\t{drone.get_current_camera_rotation()}")
    print(f"position:\t{drone.get_current_xyz()}")
    if input("Press enter to loop again or q to quit") == "q":
        break
