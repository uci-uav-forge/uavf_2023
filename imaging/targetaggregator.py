from typing import *
import csv
from .best_match import *


def color_dist(rgbA, rgbB):
    # todo - possibly measure color distance in some more sophisticated way.
    return sum((a - b) ** 2 for (a, b) in zip(rgbA, rgbB))


COLORS_TO_RGB = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'brown': (165, 42, 42),
}


def gen_color_conf(rgb, cnames):
    r0 = {
        color: color_dist(rgb, COLORS_TO_RGB[color])
        for color in cnames}

    mx = max(r0.values())
    return {k: v / mx for k, v in r0.items()}


class TargetAggregator:
    def __init__(self, targets_file):
        with open(targets_file, newline='') as tf:
            self.targets: "list[tuple[str]]" = list(map(tuple, csv.reader(tf)))
            # convert letters to uppercase and everything else to lowercase
            self.targets = [(x[0].lower(), x[1].upper(), x[2].lower(), x[3].lower()) for x in self.targets]
        self.n_targets = len(self.targets)
        self.best_conf = [-1] * self.n_targets
        self.target_coords = [None] * self.n_targets  # coord @ ith index --> target @ ith index in CSV

    def match_target_color(self, coords, letterColor, letterConf, shapeColor, shapeConf):
        letterColorConf = gen_color_conf(letterColor, [x[0].lower() for x in self.targets])
        shapeColorConf = gen_color_conf(shapeColor, [x[2].lower() for x in self.targets])

        match, score = best_match_color(self.targets, letterColorConf, letterConf, shapeColorConf, shapeConf)
        matchIndex = self.targets.index(match)

        if score > self.best_conf[matchIndex]:
            self.target_coords[matchIndex] = coords
            self.best_conf[matchIndex] = score

    def match_target(self, coords, lC, sC):
        match, score = best_match(self.targets, lC, sC)
        matchIndex = self.targets.index(match)

        if score > self.best_conf[matchIndex]:
            self.target_coords[matchIndex] = coords
            self.best_conf[matchIndex] = score

    def get_target_coords(self):
        return self.target_coords


if __name__ == '__main__':
    ta = TargetAggregator('targets.csv')
    print(ta.targets)
