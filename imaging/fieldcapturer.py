from typing import NamedTuple, Any, List

class Tile(NamedTuple):
    img: Any
    px_offset: Any
    yaw: Any
    altitude: Any
    base_coords: Any


class FieldCapturer:
    def __init__(self):
        pass
    def capture(self) -> List[Tile]:
        return []
    def update_coords(self, coords):
        pass
    def update_yaw(self, yaw):
        pass
    def update_altitude(self, altitude):
        pass