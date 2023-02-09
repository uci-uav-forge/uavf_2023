!/bin/sh
#open xterm terminal, which will run roslaunch mavros... and hold terminal open independent of original terminal
#xterm is built into linux, but gnome-terminal might be needed to open terminal over ssh
xterm -hold -e roslaunch mavros px4.launch fcu_url:="/dev/ttyPixhawk" &
#python mission script here
OPENBLAS_CORETYPE=ARMV8 python navigation/guided_mission/guided_mission.py
