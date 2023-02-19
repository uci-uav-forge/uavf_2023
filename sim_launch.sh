#!/bin/sh
# launch automation script ver.2, main difference is the use of Tmux sessions to start and run the mission commands
# such that the closing of the ssh terminal will not end the program
# make sure tmux is installed on drone

#tmux command format = tmux -d -s session_name scriptToRun_InSession.sh

# to attach to one of the tmux sessions run: "tmux attach -t session_name"
# to kill tmux session run: "tmux kill-session -t session_name"

tmux new -d -s mavros_roslaunch ./launch_scripts/start_mavros_sim.sh

tmux new -s mission_script ./launch_scripts/start_mission.sh
