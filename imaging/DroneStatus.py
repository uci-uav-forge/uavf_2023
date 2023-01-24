from dataclasses import dataclass
from collections import namedtuple

@dataclass
class DroneStatus():
    GPSCoord: tuple(float,float)
    Altitude: float
    Pitch: float
    Yaw: float
    Roll: float

    def updatecoord(self, newcoords):
        formatgps = namedtuple('Location', ['lat' , 'long' ])
        self.GPSCoord = formatgps(newcoords[0],newcoords[0])
    


def testrun():

    droneexamplestatus = DroneStatus((0,0), 10, 9,  8,  7)
    print(droneexamplestatus)


testrun()
