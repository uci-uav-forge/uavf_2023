To run both pipelines:

    1. Start the MAVROS node. `./mavros_start.sh` Refer to Navigation's README for further instructions.
    
    2. Run the imaging pipeline. `py start_pipeline.py` enter the two options for using the gopro and real location when prompted

    3. Run guided_mission. `./launch_scripts/mission.sh`