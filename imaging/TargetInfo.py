from dataclasses import dataclass, InitVar
from collections import namedtuple
from typing import Optional

@dataclass
class TargetInfo:
    Shape: str
    Colors: tuple([str, str])
    Letter: str
    CenterCoord: tuple([float,float])
    CalcGPSCoord: InitVar[Optional[tuple]] = None


    def updateGPS(self, missingcoord):
        self.CalcGPSCoord = missingcoord



if __name__ == '__main__':
    targetexampleinfo = TargetInfo("triangle",("blue","black"),"A", (4,6))
    print(targetexampleinfo)
    targetexampleinfo.updateGPS((500,200))
    print(targetexampleinfo)
    print(targetexampleinfo.CalcGPSCoord)
