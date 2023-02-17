import rospy
from navigation.guided_mission.run_mission import nav_main, Localizer
from queue import Queue
import imaging.src.main as imaging
import threading
'''
runs imaging and navigation code concurrently.
'''

# contains objects used by both navigation and imaging code.
class Interop:
    def __init__(self, loc, mq):
        self.localizer = loc
        self.mq = mq

if __name__ == '__main__':
    rospy.init_node('uavf')

    io = Interop(Localizer(), Queue())

    navThread = threading.Thread(target = nav_main, args = (io,))

    navThread.run()

    imaging.main(io)


