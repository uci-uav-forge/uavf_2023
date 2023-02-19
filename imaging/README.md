Codebase for the Imaging pipeline.
Main run loop inside `yolo_main.py`

### GoPro Camera setup:
Install the GoPro webcam utility from [here](https://github.com/jschmid1/gopro_as_webcam_on_linux).
    This allows the GoPro to work with Ubuntu.
    Most recent command to install:
        $ sudo su -c "bash <(wget -qO- https://cutt.ly/PjNkrzq)" root

Once package installed, run the following commands:
    $ sudo apt-get install v4l2loopback-source module-assistant
    $ sudo module-assistant auto-install v4l2loopback-source
    $ sudo depmod
    $ sudo modprobe v4l2loopback

To run the camera with Imaging pipeline:
    $ sudo gopro webcam -a -n
    Appending `-f narrow` enables narrow field of view for better zoom

