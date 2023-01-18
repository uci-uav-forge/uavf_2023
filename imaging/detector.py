'''
Manages ML models for location, shape/color detection
and runs it on tile input.
'''

from typing import *

class Target(NamedTuple):
    color: Any
    letter: Any
    shape: Any
    center_pix: Tuple[float, float]
    center_coords: Optional[Tuple[float, float]]

class Detector:
    def __init__(self):
        # owns the ml models used for detection.
        pass

    def detect(self, image: Any) -> List[Target]:
        return None

