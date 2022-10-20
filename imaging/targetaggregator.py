from typing import *
import detector

class TargetAggregator:
    def __init__(self):
        self.targets_by_attrs = {}
    def add_target(self, target: detector.Target):
        pass
    def get_targets(self) -> List[detector.Target]:
        return list(self.targets_by_attrs.values())