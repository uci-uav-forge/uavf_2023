from typing import *
import detector

class FrameSummary:
    def __init__(self):
        self.targets = []
    def register_target(self, target: detector.Target):
        self.targets.append(target)
    def get_targets(self) -> List[detector.Target]:
        return self.targets