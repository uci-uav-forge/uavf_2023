from typing import *
import statistics
from TargetInfo import TargetInfo
import collections

class TargetAggregator:
    # subjective value of .00001 degrees = 1.11m; TODO adjust based on empirical variance
    GPS_EPSILON = 0.00007

    def __init__(self):
        # A map from target attributes to lists of target groups - 
        # each target group is a list of positions found which are close to each other.
        # We will aggregate each group somehow when we finalize coordinates
        # ex: {(square,red,W,green) : [[(50,50), (50.004,50), ...],[(100,100)], ... ] }
        self.targets_by_attrs = collections.defaultdict(list)

    def add_target(self, target: TargetInfo):
        assert target.CalcGPSCoord != None

        attribute_key = (target.Shape, target.Colors[0], target.Letter, target.Colors[1])

        matching_target_groups = self.targets_by_attrs[attribute_key]

        for target_group in matching_target_groups:
            for coord in target_group:
                if abs(coord[0] - target.CalcGPSCoord[0]) <= TargetAggregator.GPS_EPSILON and \
                    abs(coord[1] - target.CalcGPSCoord[1]) <= TargetAggregator.GPS_EPSILON:
                    # Return early if we find a pre-existing group this measurement is close to, adding it to that group.
                    target_group.append(target.CalcGPSCoord)
                    return

        # This attribute combination is far from all others like it. Start a new group.
        self.targets_by_attrs[attribute_key].append([target.CalcGPSCoord])
    
    def get_targets(self) -> dict[Tuple, List[Tuple]]:
        # returns map from target attributes to list of median positions
        result = {attr: [(statistics.median(coord[0] for coord in group), 
                          statistics.median(coord[1] for coord in group))
                        for group in groups]
                    for attr, groups in self.targets_by_attrs.items()}
        return result



if __name__ == '__main__':
    agg = TargetAggregator()

    testshape_1 = TargetInfo('triangle', ('red','black'), 'Q', (4,6))
    testshape_2 = TargetInfo('square', ('green','white'), 'X', (4,6))

    testshape_1.updateGPS((4,6))
    testshape_2.updateGPS((4,6))

    agg.add_target(testshape_1)
    agg.add_target(testshape_2)

    assert agg.get_targets() == {('triangle','red','Q','black') : [(4,6)], ('square','green','X','white') : [(4,6)]}, \
        "Correctly returns inputted targets"

    TargetAggregator.GPS_EPSILON = 10

    testshape_3 = TargetInfo('triangle', ('red','black'), 'Q', (4,10), (4,10))
    testshape_3.updateGPS((4,10))
    agg.add_target(testshape_3)


    assert agg.get_targets() == {('triangle','red','Q','black') : [(4,8)], ('square','green','X','white') : [(4,6)]}, \
        "Combines close targets"

    testshape_4 = TargetInfo('triangle', ('red','black'), 'Q', (4,30), (4,30))
    testshape_4.updateGPS((4,30))
    agg.add_target(testshape_4)

    assert agg.get_targets() == {('triangle','red','Q','black') : [(4,8), (4,30)], ('square','green','X','white') : [(4,6)]}, \
        "Doesn't combine far targets"
