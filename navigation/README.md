To SSH into Jetson Nano (test quad):
	IP address 172.20.10.2 on Omar's Iphone hotspot
	

For PX4: 
	Actual Flights:
		Set up telemetry on companion computer (plug in cube to usb port):
			(do the udev rule stuff)
			https://docs.px4.io/main/en/companion_computer/pixhawk_companion.html
	
		Run irl drone MAVROS node:
			roslaunch mavros px4.launch fcu_url:="/dev/ttyPixhawk"
		
		Append this command before running python scripts:
			OPENBLAS_CORETYPE=ARMV8 
	
	Simulation:
		Run QGroundControl:
			$ ./QGroundControl.AppImage

		Run simulation:
			cd into PX4-Autopilot
			$ sudo make px4_sitl_default gazebo

		To launch simulation MAVROS node:
			$ roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"

	To rebuild catkin workspace:
		$ cd ~/catkin_ws
		$ catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m

	Installing ROS and MAVROS (Ubuntu 20):
		https://docs.google.com/document/d/1iDZaO9g8QdiUE_a3mRM_urEm7MaIqPoOlQ1OcymD9Dk/edit

	Running PX4 simulation:
		https://docs.google.com/document/d/1XeKneJUelfyDAyo95s333sQnyqmPuVQOimS5eBxL6Bw/edit


Setting up Ardupilot, MAVProxy, Guided Mode:

	1. Ardupilot and MAVProxy 
		Clone ardupilot into home directory.
		$ git clone https://github.com/ArduPilot/ardupilot.git
		$ cd ardupilot

		$ Tools/environment_install/install-prereqs-ubuntu.sh -y
		$ . ~/.profile

		$ git checkout Copter-4.3
		$ git submodule update --init --recursive		
		cd into ardupilot/ArduCopter
		$ sim_vehicle.py -w

		If you run into error (no pymavlink):
			$ python -m pip install pymavlink
			cd into ardupilot/Tools/autotest
			$ python sim_vehicle.py --map --console
			$ python sim_vehicle.py -v ArduCopter --map --console
				Fix any error messages and keep repeating until ArduCopter launches!
			cd into ardupilot/ArduCopter
			$ python sim_vehicle.py -w
				ArduCopter should launch now!

	2. ArduCopter ROS-Gazebo Simulation
		$ sudo apt install xterm
		$ sudo apt install ros-noetic-gazebo-ros ros-noetic-gazebo-plugins
		cd into catkin_ws/src
		$ git clone https://github.com/Intelligent-Quads/iq_sim.git
		cd into catkin_ws
		$ catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m

		Convenience script for launching ArduCopter simulator:
			$ cp ~/catkin_ws/src/iq_sim/scripts/startsitl.sh ~
			Now you can start Ardupilot sitl with:
				$ ~/startsitl.sh
		
		Launch the gazebo world in a new terminal:
			cd into catkin_ws
			$ source devel/setup.bash
				If a launch file isnt working, try sourcing the workspace first
			$ roslaunch iq_sim droneOnly.launch
				It will take a while to launch the first time!
		
		Start MAVROS node in a new terminal:
			$ roslaunch iq_sim apm.launch
				Launches MAVROS node.
		
		Test your simulation!
			Open up Gazebo and the terminal where you launched startsitl.sh
			Press 'enter', the terminal should now say 'STABILIZE>'
			Check MAVProxy console while sending commands to see the drone status.
				$ mode guided 
				$ arm throttle 
				$ takeoff 15
				$ mode land
	
	3. Guided Mode
		cd into catkin_ws/src
		$ git clone https://github.com/Intelligent-Quads/iq_gnc.git
		$ cd iq_gnc/scripts
			Scripts for dynamic control using guided mode.
			We can use these as a template for our own scripts.
		$ cd ../src/iq_gnc/
			py_gnc_functions.py has methods that abstract the guided mode interface for us.
			We can use these methods or use them as a template for our own.
			I have copied these python files into our own repository for convenience. 
	
	4. To summarize...
		$ ~/startsitl.sh
			starts the ArduCopter sitl and MAVProxy ground control station.
		$ roslaunch iq_sim droneOnly.launch
			launches the gazebo simulation.
		$ roslaunch iq_sim apm.launch
			starts MAVROS node.

	Source: https://github.com/Intelligent-Quads/iq_tutorials


Setting Up Nvidia TX2 as Companion Computer:

	1. https://ardupilot.org/dev/docs/companion-computer-nvidia-tx2.html#companion-computer-nvidia-tx2

	2. Install APSync 
		https://ardupilot.org/dev/docs/apsync-intro.html#apsync-intro-installing-apsync
	
	3. Connect MAVROS
		https://ardupilot.org/dev/docs/ros-connecting.html