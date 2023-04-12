from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
import rospy
import message_filters
from tf.transformations import euler_from_quaternion
import numpy as np
import time
from datetime import datetime
from math import degrees

class GPS_Attitude_Recorder():

    def __init__ (self):
        # write new file 
        now = datetime.now()
        dt_string = str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'_'+str(now.hour)+':'+str(now.minute)
        #dt_string = datetime.now.strftime("%d/%m/%Y %H:%M:%S")
        self.file = open("telemetry_" + dt_string + ".txt", "w+")
        
        # vehicle attitude stats and counter
        self.min = np.empty(3, dtype=np.float64)
        self.max = np.empty(3, dtype=np.float64)
        self.avg = np.empty(3, dtype=np.float64)
        self.count = 1

        # initialize node and subs, sync subs into 1 callback
        #rospy.init_node("GPS_telemetry_subscriber", anonymous=True)
        
        rospy.init_node("GPS_Attitude_subscriber_node")
        pose_sub = rospy.Subscriber(
            name='/mavros/local_position/pose', data_class=PoseStamped, queue_size=1, callback=self.callback,
        )
        
        # record start time and start timer
        self.start = time.time()
        self.timer = self.start
        rospy.spin() 

        # write vehicle attitude stats at the end
        self.file.write(
            '\n' + 'Min Pitch, Roll, Yaw: ' + str(self.min) + '\n' +\
                   'Max Pitch, Roll, Yaw: ' + str(self.max) + '\n' +\
                   'Avg Pitch, Roll, Yaw: ' + str(self.avg) + '\n' + str(self.count)
        )
        self.file.close()    


    def callback(self, pose):
        # time step at least 2 seconds
        curr = time.time()
        if (curr - self.timer >= 1):
            self.timer = time.time()

            # get euler angles from quaternion
            orient = np.array(
                [pose.pose.orientation.x, pose.pose.orientation.y,
                pose.pose.orientation.z, pose.pose.orientation.w]
            )
            roll, pitch, yaw = euler_from_quaternion(orient)
            att_arr = np.asarray((
                degrees(roll), degrees(pitch), degrees(yaw)
            ))

            print('Roll: ' + str(degrees(roll)))
            print('Pitch: ' + str(degrees(pitch)))
            print()

            # update vehicle attitude stats
            for i in range(3):
                if att_arr[i] < self.min[i]:
                    self.min[i] = att_arr[i]
                elif att_arr[i] > self.max[i]:
                    self.max[i] = att_arr[i]
            self.avg = (self.count*self.avg + att_arr) / (self.count + 1)
            self.count += 1

            # write telemetry info
            info = 'Position: (' + str(round(pose.pose.position.x,3)) + ', ' + str(round(pose.pose.position.y,3)) +\
                    ', ' + str(round(pose.pose.position.z,3)) + ')    Time: ' + str(round(curr-self.start,3)) + ' s\n'
            self.file.write(info)


if __name__ == "__main__":
    recorder = GPS_Attitude_Recorder()
    