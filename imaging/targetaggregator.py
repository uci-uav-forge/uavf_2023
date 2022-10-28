from typing import *
import detector
import TargetInfo

#want to return a list of gps coordinates, mostlikely a list of tuples
class TargetAggregator:
    def __init__(self):
        self.targets_by_attrs = {} 


# ex: {(square,red,W,green) : [[(50,50), (80,90), ...],[], ... ] }

    def add_target(self, target: TargetInfo): #included comparison and addition
        if (len(self.targets_by_attrs)==0):
            attributekey = (target.Shape,target.Colors[0],target.Letter,target.Colors[1])
            self.targets_by_attrs[attributekey] = [[target.CalcGPSCoord]]

        else:
            for eachtarget in self.targets_by_attrs:
                if (target.Shape, target.Colors[0], target.Letter, target.Colors[1] ) == (eachtarget):
                    for eachgpsgroup in self.targets_by_attrs[eachtarget]:
                        # iterating over a list of list of tuple so [[(59,30),(50,38),] ]
                        # so eachgpsgroup is the list inside of the big list 
                        for eachcoord in eachgpsgroup:
                        # value is subjective though .00001 degrees = 1.11m; can be changed the range
                            if (abs(eachcoord[0] - target.CalcGPSCoord[0])<= 0.00007) & (abs(eachcoord[1] - target.calcGPSCoord[1])<= 0.00007):
                                eachgpsgroup.append(target.CalcGPSCoord)
                                target.CalcGPSCoord = None
                            else:
                                self.targets_by_attrs[eachtarget].append([target.CalcGPSCoord])
                                target.CalcGPSCoord = None

            if (target.CalcGPSCoord != None):
                attributekey = (target.Shape,target.Colors[0],target.Letter,target.Colors[1])
                self.targets_by_attrs[attributekey] = [[target.CalcGPSCoord]]

        
    def get_targets(self) -> List[detector.Target]:
        return list(self.targets_by_attrs.values())
    
    def finalize_gps(self):
        for eachattrs in self.targets_by_attrs:
            for eachgroup in self.targets_by_attrs[eachattrs]:
                sortlat, sortlong = [],[]
                for eachcoord in eachgroup:
                    sortlat.append(eachcoord[0])
                    sortlat.append(eachcoord[1])
                sortlat.sort()
                sortlong.sort()
                median = sortlat.length() // 2
                eachgroup = [(sortlat[median],sortlong[median])]
        #expected result (shape,shapecolor,letter,lettercolor):[[(lat1,long1)],[(lat2,long2)], etc.]


        


# order: cmp target info to existing targetaggregator,targets_by_attrs 
# returns the target if it needs to be added to the list then add it to the target aggregator attributes


