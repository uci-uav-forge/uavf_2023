from math import atan2, asin, degrees

import rospy
from scipy.spatial.transform import Rotation
import numpy as np
from tf import transformations
from navigation.guided_mission.py_gnc_functions import gnc_api

rospy.init_node("drone_GNC", anonymous=True)
drone = gnc_api()
drone.initialize_local_frame()

while 1:
    angles = drone.get_current_camera_rotation()
    heading = (angles[0]+angles[2] + 360)%360 
    print(f"camera angles:\t{angles}")
    print(f"heading: {heading}")
    print(f"position:\t{drone.get_current_xyz()}")
    if input("Press enter to loop again or q to quit") == "q":
        break
