from collections import defaultdict
import argparse

SHAPE_OFFSETS = defaultdict( lambda: [0,0,0],
        {
            'Circle': [0,0,0],
            'Heptagon': [0,0,0],
            'Halfcircle': [0,0,0],
            'Hexagon': [0,0,0],
            'Octagon': [0,0,0],
            'Pentagon': [0,0,0],
            'Plus': [0,0,0],
            'QuarterCircle': [0,0,0],
            'Rectangle': [0,0,0],
            'Square': [0,0,0],
            'Star': [0,0,0],
            'Trapezoid': [0,0,0],
            'Triangle': [0,0,0]
        })
SHAPE_SCALES = defaultdict( lambda: 1,
        {
            'Circle': 1,
            'Heptagon': 1,
            'Halfcircle': 1,
            'Hexagon': 1,
            'Octagon': 1,
            'Pentagon': 1,
            'Plus': 1,
            'QuarterCircle': 1,
            'Rectangle': 1,
            'Square': 1,
            'Star': 1,
            'Trapezoid': 1,
            'Triangle': 1
        })

class BlenderArgParse(argparse.ArgumentParser):
    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index('--')
            return sys.argv[idx+1 :]
        except ValueError as e:
            return []
    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())
import sys
argv = sys.argv
argv = argv[argv.index('--')+1 :]
PARSER = BlenderArgParse()
#required ARGUMENTS
PARSER.add_argument('--dir', type=str, help='directory of dataset', required=True)
PARSER.add_argument('--backgnd', type=str, help='directory of background', required=True)
PARSER.add_argument('--shape_dir', type=str, help='directory of shapes', required=True)
PARSER.add_argument('--alpha_dir', type=str, help='directory of alphanum', required=True)
PARSER.add_argument('--n', type=int, help='number of image data to generate', required=True)

#Optional Arguments, to manipulate type of data
PARSER.add_argument('--min_width', help='minimum width of target in meters',type=float, 
        default=.3,required=False)
PARSER.add_argument('--max_width', help='maximum width of target in meters', type=float,
        default=.4,required=False)
PARSER.add_argument('--resolution', help='images generated are nxn. you are picking n', type=int,
        default=512, required=False)
PARSER.add_argument('--upper_pitch', help='Blender Camera Pitch Upper Bound (degrees)', type=float,
        default=30, required=False)
PARSER.add_argument('--lower_pitch', help='Blender Camera Pitch Lower Bound (degrees)', type=float,
        default=-30, required=False)
PARSER.add_argument('--upper_roll', help='Blender Camera Roll Upper Bound (degrees)', type=float,
        default=30, required=False)
PARSER.add_argument('--lower_roll', help='Blender Camera Roll Lower Bound (degrees)', type=float,
        default=-30, required=False)
PARSER.add_argument('--focal_len', help='Blender Camera Focal Length (mm)', type=float, 
        default=205, required=False)
PARSER.add_argument('--upper_altitude', help='Blender Camera altitude (meters) upper bound', type=float,
        default=31, required=False)
PARSER.add_argument('--lower_altitude', help='Blender Camera altitude (meters) lower bound', type=float,
        default=30, required=False)
PARSER.add_argument('--alphascale', help='ratio: Alphanumeric width / Target Width ', type=float,
        default=0.35, required=False)
OPTS = PARSER.parse_args()

