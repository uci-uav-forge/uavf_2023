'''
Takes picture of the field below the drone and returns tiles from the picture which
are tagged with metadata including pixel offest and drone position.
'''
from typing import NamedTuple, Any, List

class DroneStatus(NamedTuple):
    yaw: Any
    altitude: Any
    coords: Any

class Tile(NamedTuple):
    img: Any
    px_offset: Any

class CaptureResult(NamedTuple):
    drone_status: DroneStatus
    tiles: List[Tile]
    img: Any

class FieldCapturer:
    def __init__(self):
        pass
    def capture(self) -> CaptureResult:
        pass
    def update_coords(self, coords):
        pass
    def update_yaw(self, yaw):
        pass
    def update_altitude(self, altitude):
        pass