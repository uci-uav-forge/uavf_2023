import requests
import numpy as np
import cv2 as cv

class GoProCamera:
    def __init__(self):
        # uses https://gopro.github.io/OpenGoPro/http_2_0
        self.url = "http://172.23.157.51:8080"
        requests.get(f"{self.url}/gopro/camera/control/wired_usb?p=1")# enable wired control
    
    def get_image(self) -> cv.Mat:
        requests.get(f"{self.url}/gopro/camera/shutter/start")
        busy=True
        while busy:
            statuses: dict = requests.get(f"{self.url}/gopro/camera/state").json()["status"]
            busy = statuses['8'] or statuses['10']
        media_list: dict = requests.get(f"{self.url}/gopro/media/list").json()['media']
        file_name = media_list[0]['fs'][-1]['n']
        img_bytes = requests.get(f"{self.url}/videos/DCIM/100GOPRO/{file_name}").content
        return cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
    
if __name__=="__main__":
    cam = GoProCamera()
    img = cam.get_image()
    cv.imwrite("gopro_img.png", img)