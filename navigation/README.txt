To launch MAVROS node:
	$ roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"

To rebuild catkin workspace:
	$ cd ~/catkin_ws
	$ catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m