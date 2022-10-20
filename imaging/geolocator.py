'''
Modifies targets by attaching gps position
'''
import detector
import fieldcapturer

class Geolocator:
    def __init__(self):
        pass
    
    def locate(self, target: detector.Target, drone_status: fieldcapturer.DroneStatus) -> detector.Target:
        pass