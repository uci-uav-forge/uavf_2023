#!/bin/sh
# shell script for launching mavros
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
