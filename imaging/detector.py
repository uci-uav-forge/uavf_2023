from typing import *
import fieldcapturer

class Target(NamedTuple):
    color: Tuple[float,float,float]
    letter: str
    shape: str
    center_pix: Tuple[float, float]
    center_coords: Optional[Tuple[float, float]]

class Detector:
    def __init__(self):
        pass

    def detect(self, tile: fieldcapturer.Tile) -> Target:
        return None

