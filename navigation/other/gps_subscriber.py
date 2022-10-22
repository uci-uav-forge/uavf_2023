import rospy
import message_filters
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
import numpy as np
import time

class GPS_Sub():

    def __init__ (self):
        #reset file contents
        self.file = open("GPS_record.txt", "w")
        self.file.close()
        self.file = open("GPS_record.txt", "w")
        
        self.min = np.empty(3, dtype=np.float64)
        self.max = np.empty(3, dtype=np.float64)
        self.avg = np.empty(3, dtype=np.float64)
        self.count = 1
        
        rospy.init_node("GPS_Attitude_subscriber_node")
        gps_sub = message_filters.Subscriber(
            "/mavros/global_position/raw/fix", NavSatFix)
        pose_sub = message_filters.Subscriber(
            '/mavros/local_position/pose', PoseStamped)
        sync = message_filters.ApproximateTimeSynchronizer(
                        [gps_sub, pose_sub], 10, 0.1)
        sync.registerCallback(self.callback)
        
        self.start = time.time()
        self.timer = self.start
        rospy.spin()       


    def callback(self, gps, pose):
        curr = time.time()
        if (curr - self.timer >= 2):
            self.timer = time.time()

            orient = np.array([pose.pose.orientation.x, pose.pose.orientation.y,
                            pose.pose.orientation.z, pose.pose.orientation.w])
            roll, pitch, yaw = euler_from_quaternion(orient)
            att_arr = np.asarray((roll, pitch, yaw))

            for i in range(3):
                if att_arr[i] < self.min[i]:
                    self.min[i] = att_arr[i]
                elif att_arr[i] > self.max[i]:
                    self.max[i] = att_arr[i]
            self.avg = (self.count*self.avg + att_arr) / (self.count + 1)
            self.count += 1

            info = 'Lat: '+str(gps.latitude) + '    Long: '+str(gps.longitude) +\
                '    Alt: '+str(gps.altitude) + '    Time: '+str(round(curr-self.start, 3))+' s'+ '\n'
            self.file.write(info)
            print('minimum pitch, roll, yaw: ' + str(self.min))
            print('maxo,i, pitch, roll, yaw: ' + str(self.max))
            print('average pitch, roll, yaw: ' + str(self.avg))
            print()


if __name__ == "__main__":
    new_sub = GPS_Sub()
