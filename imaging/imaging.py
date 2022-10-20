import detector
import fieldcapturer
import geolocator
import targetaggregator


class Imager:
    def __init__(self):
        self.detector = detector.Detector()
        self.field_capturer = fieldcapturer.FieldCapturer()
        self.geolocator = geolocator.Geolocator()
        self.target_aggregator = targetaggregator.TargetAggregator()
    def capture(self):
        capture_result = self.field_capturer.capture()

        frame_targets = []

        # todo, this work and work below is parallelizable
        for tile in capture_result.tiles:
            result = self.detector.detect(tile)
            if result != None:
                frame_targets.append(result)

        frame_targets = [
            target._replace(
                center_coords=
                    self.geolocator.locate(
                        target.center_pix, 
                        capture_result.drone_status)) 
                for target in frame_targets]
        
        for target in frame_targets:
            self.target_aggregator.add_target(target)
        