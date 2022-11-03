'''
Takes picture of the field below the drone and returns tiles from the picture which
are tagged with metadata including pixel offest and drone position.
'''

from typing import NamedTuple, Any, List
from open_gopro import GoPro, Params
import cv2

class DroneStatus(NamedTuple):
    yaw: Any
    altitude: Any
    coords: Any

class FieldCapturer:
    def __init__(self):
        self.photo_counter = 0

        self.gp = GoPro()
        self.gp.__enter__()
        # Configure settings to prepare for photo
        if self.gp.is_encoding:
            self.gp.ble_command.set_shutter(Params.Toggle.DISABLE)
        self.gp.ble_setting.video_performance_mode.set(Params.PerformanceMode.MAX_PERFORMANCE)
        self.gp.ble_setting.max_lens_mode.set(Params.MaxLensMode.DEFAULT)
        self.gp.ble_setting.camera_ux_mode.set(Params.CameraUxMode.PRO)
        self.gp.ble_command.set_turbo_mode(False)
        assert self.gp.ble_command.load_preset_group(Params.PresetGroup.PHOTO).is_ok
        
    def capture(self) -> Any:
        filename = f"photo{self.photo_counter}.jpg"
        self.photo_counter += 1

        # Get the media list before
        media_set_before = set(x["n"] for x in self.gp.wifi_command.get_media_list().flatten)
        # Take a photo
        print("Capturing a photo...")
        assert self.gp.ble_command.set_shutter(Params.Toggle.ENABLE).is_ok

        # Get the media list after
        media_set_after = set(x["n"] for x in self.gp.wifi_command.get_media_list().flatten)
        # The photo (is most likely) the difference between the two sets
        photo = media_set_after.difference(media_set_before).pop()
        # Download the photo
        print("Downloading the photo...")
        self.gp.wifi_command.download_file(camera_file=photo, local_file=filename)
        print(f"Success!! :smiley: File has been downloaded to {filename}")
        return cv2.imread(filename)

    def get_drone_status(self) -> DroneStatus:
        pass
    def update_coords(self, coords):
        pass
    def update_yaw(self, yaw):
        pass
    def update_altitude(self, altitude):
        pass

if __name__ == '__main__':
    fc = FieldCapturer()
    img = fc.capture()
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()