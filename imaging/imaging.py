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
        capture_image = self.field_capturer.capture()
        drone_status = self.field_capturer.get_drone_status()

        frame_targets = self.detector.detect(capture_image)
        
        for target in frame_targets:
            target.center_coords = \
                self.geolocator.locate(target.center_pix, drone_status)
        
        return frame_targets #???
'''
        for target in frame_targets:
            self.target_aggregator.add_target(target)
'''
        