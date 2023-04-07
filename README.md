To run both pipelines:

    1. Start the MAVROS node. Refer to Navigation's README for further instructions.
    
    2. Start camera stream:
        $ sudo gopro webcam -a -n
            Append ``-f narrow` for a narrower field of view

            If GoPro webcam utility not installed, refer to Imaging's README

            Run the next commend only when the dev port is listed for camera initialization
    
    3. Run the master script that commences both Imaging and Navigation pipeline
        $ python master.py

All three commands must be running simultaneously


https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics
