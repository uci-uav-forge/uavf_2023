import requests
import numpy as np
import cv2 as cv

class GoProCamera:
    def __init__(self, fov_mode = "broken_currently"):
        '''
        TODO: figure out why changing the fov mode doesn't actually do anything on the camrea.
        fov_mode: "narrow", "superview", "wide", "linear"
        https://community.gopro.com/s/article/HERO10-Black-Digital-Lenses-FOV-Informations?language=en_US

        Sorted by zoom level (least FOV and highest zoom first):
        1. narrow
        2. linear
        3. wide
        4. superview
        '''
        if fov_mode!="broken_currently":
            print("Warning: FOV mode currently doesn't change anything.")
        # uses https://gopro.github.io/OpenGoPro/http_2_0
        self.url = "http://172.23.157.51:8080"
        self._wait_on_busy()
        requests.get(f"{self.url}/gopro/camera/control/wired_usb?p=1") # enable wired control
        # self._wait_on_busy()
        # self._fov_dict = {
        #     "narrow": 19,
        #     "superview": 100,
        #     "wide": 100,
        #     "linear": 102
        # }
        # requests.get(f"{self.url}/gopro/camera/setting?setting=122&option={self._fov_dict[fov_mode]}") # set photo mode

    def set_fov_mode(self, fov_mode: str):
        '''
        See constructor for valid fov_modes
        '''
        self._wait_for_busy()
        requests.get(f"{self.url}/gopro/camera/setting?setting=122&option={self._fov_dict[fov_mode]}")

    def _wait_on_busy(self):
        '''
        Waits until the camera is no longer busy (see here for details: https://gopro.github.io/OpenGoPro/http_2_0#sending-commands)
        '''
        busy = True
        while busy:
            statuses: dict = requests.get(f"{self.url}/gopro/camera/state").json()["status"]
            busy = statuses['8'] or statuses['10']
    
    def get_image(self) -> cv.Mat:
        self._wait_on_busy()
        requests.get(f"{self.url}/gopro/camera/shutter/start")
        self._wait_on_busy()
        media_list: dict = requests.get(f"{self.url}/gopro/media/list").json()['media']
        file_name = media_list[0]['fs'][-1]['n']
        img_bytes = requests.get(f"{self.url}/videos/DCIM/100GOPRO/{file_name}").content
        return cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
    
if __name__=="__main__":
    cam = GoProCamera()
    for fov_mode in ["narrow", "linear", "wide", "superview"]:
        img = cam.get_image()
        cv.imwrite(f"gopro_{fov_mode}.png", img)