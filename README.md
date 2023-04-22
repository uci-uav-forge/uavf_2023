To run both pipelines:

    1. Start the MAVROS node. `./mavros_start.sh` Refer to Navigation's README for further instructions.
    
    2. Run the imaging pipeline. `py start_pipeline.py` enter the two options for using the gopro and real location when prompted

    3. Run guided_mission. `./launch_scripts/mission.sh`

Right now the payload code is under `navigation/guided_mission/servo_controller.py` running that with a command like `py navigation/guided_mission/servo_controller.py` will let you open and close the motors.

`navigation/guided_mission/guided_mission.py` is the main script to run. If you want to test it without doing the payload drop you can add env vars to the launch script like so: `MOCK_PAYLOAD=y ./launch_scripts/mission.sh`. You can change the `wait_for_imaging` keyword argument of `mission_loop` in the main function to choose whether or not it waits for imaging to be done.

To kill any of the processes early, do ctrl+backslash(`\\`). This is necessary because a lot of our scripts don't respond to the normal `ctrl+c` interrupt signal.

