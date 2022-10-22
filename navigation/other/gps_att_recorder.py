from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
import rospy
import message_filters
from tf.transformations import euler_from_quaternion
import numpy as np
import time

class GPS_Attitude_Recorder():

    def __init__ (self):
        #reset file contents
        self.file = open("GPS_record.txt", "w")
        self.file.close()
        self.file = open("GPS_record.txt", "w")
        # vehicle attitude stats and counter
        self.min = np.empty(3, dtype=np.float64)
        self.max = np.empty(3, dtype=np.float64)
        self.avg = np.empty(3, dtype=np.float64)
        self.count = 1
        # initialize node and subs, sync subs into 1 callback
        rospy.init_node("GPS_Attitude_subscriber_node")
        gps_sub = message_filters.Subscriber(
            "/mavros/global_position/raw/fix", NavSatFix)
        pose_sub = message_filters.Subscriber(
            '/mavros/local_position/pose', PoseStamped)
        sync = message_filters.ApproximateTimeSynchronizer(
                        [gps_sub, pose_sub], 20, 2)
        sync.registerCallback(self.callback)
        # record start time and start timer
        self.start = time.time()
        self.timer = self.start
        rospy.spin() 
        # write vehicle attitude stats at the end
        self.file.write(
            '\n' + 'Min Pitch, Roll, Yaw: ' + str(self.min) + '\n' +\
                   'Max Pitch, Roll, Yaw: ' + str(self.max) + '\n' +\
                   'Avg Pitch, Roll, Yaw: ' + str(self.avg) + '\n'
        )      


    def callback(self, gps, pose):
        # time step at least 2 seconds
        curr = time.time()
        if (curr - self.timer >= 2):
            self.timer = time.time()
            # get euler angles from quaternion
            orient = np.array([pose.pose.orientation.x, pose.pose.orientation.y,
                            pose.pose.orientation.z, pose.pose.orientation.w])
            roll, pitch, yaw = euler_from_quaternion(orient)
            att_arr = np.asarray((roll, pitch, yaw))
            # update vehicle attitude stats
            for i in range(3):
                if att_arr[i] < self.min[i]:
                    self.min[i] = att_arr[i]
                elif att_arr[i] > self.max[i]:
                    self.max[i] = att_arr[i]
            self.avg = (self.count*self.avg + att_arr) / (self.count + 1)
            self.count += 1
            # write GPS info
            info = 'Lat: '+str(round(gps.latitude,6)) + '    Long: '+str(round(gps.longitude,6)) +\
                '    Alt: '+str(round(gps.altitude,6)) + '    Time: '+str(round(curr-self.start, 4))+' s'+ '\n'
            self.file.write(info)


if __name__ == "__main__":
    recorder = GPS_Attitude_Recorder()
    