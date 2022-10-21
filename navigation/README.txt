To launch MAVROS node:
	$ roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"

To rebuild catkin workspace:
	$ cd ~/catkin_ws
	$ catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m


Installing ROS and MAVROS (Ubuntu 20):
	https://docs.google.com/document/d/1iDZaO9g8QdiUE_a3mRM_urEm7MaIqPoOlQ1OcymD9Dk/edit

Running PX4 simulation:
	https://docs.google.com/document/d/1XeKneJUelfyDAyo95s333sQnyqmPuVQOimS5eBxL6Bw/edit
