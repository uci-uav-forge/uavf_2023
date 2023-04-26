#!/bin/sh

#shell script for running guided_mission python file
#NOTE: cd is set such that this script is only possible to run in the ~/uavf_2023/ directory, ideally by being called by launch_ver2.sh 
#attempting to call this script from any other directory will fail

#cd navigation
#OPENBLAS_CORETYPE=ARMV8 python guided_mission.py
python3 -m navigation.guided_mission.guided_mission
