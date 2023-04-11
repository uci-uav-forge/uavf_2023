from math import atan2, asin, degrees

import rospy
from scipy.spatial.transform import Rotation
import numpy as np

from navigation.guided_mission.py_gnc_functions import gnc_api

rospy.init_node("drone_GNC", anonymous=True)
drone = gnc_api()
#drone.initialize_local_frame()

while 1:
    angles = drone.get_current_pitch_roll_yaw()
    print(angles)
    continue
    q0, q1, q2, q3 = drone.get_orientation_quaternion_wxyz() 
    if np.linalg.norm([q0,q1,q2,q3])==0:
        print("zero norm quaternion")
        continue
    rot = Rotation.from_quat([[q1,q2,q3,q0]])
    rot = rot.as_euler('zxz', degrees=True)
    print(f"scipy answer:\n{rot}")
    phi = atan2((2*(q0*q1+q2*q3)), (1-2*(pow(q1, 2)+pow(q2, 2)))) # roll

    theta = asin(2*(q0*q2-q1*q1)) # pitch

    psi = atan2((2 * (q0 * q3 + q1 * q2)),
                (1 - 2 * (pow(q2, 2) + pow(q3, 2)))) # yaw
    print("manual math answer:")
    print(degrees(phi), degrees(theta), degrees(psi))
    if input("Press enter to loop again or q to quit") == "q":
        break