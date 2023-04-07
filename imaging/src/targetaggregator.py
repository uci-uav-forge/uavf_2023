from typing import *
import csv
from .best_match import *
import webcolors


def color_dist(rgbA, rgbB):
    # TODO: measure color distance in some more sophisticated way.
    return sum((a - b) ** 2 for (a, b) in zip(rgbA, rgbB))


def gen_color_conf(rgb, cnames):
    r0 = {
        color: color_dist(rgb, webcolors.name_to_rgb(color))
        for color in cnames}

    mx = max(r0.values())
    return {k: v / mx for k, v in r0.items()}


class TargetAggregator:
    def __init__(self, targets_file):
        with open(targets_file, newline='') as tf:
            self.targets = list(map(tuple, csv.reader(tf)))
        self.n_targets = len(self.targets)
        self.best_conf = [-1] * self.n_targets
        self.target_gps = [None] * self.n_targets  # corresponds to list of targets in CSV

    def match_target_color(self, gps, letterColor, letterConf, shapeColor, shapeConf):
        letterColorConf = gen_color_conf(letterColor, [x[0] for x in self.targets])
        shapeColorConf = gen_color_conf(shapeColor, [x[2] for x in self.targets])
        match, score = best_match_color(self.targets, letterColorConf, letterConf, shapeColorConf, shapeConf)
        matchIndex = self.targets.index(match)

        if score > self.best_conf[matchIndex]:
            self.target_gps[matchIndex] = gps
            self.best_conf[matchIndex] = score

    def match_target(self, gps, lC, sC):
        match, score = best_match(self.targets, lC, sC)
        matchIndex = self.targets.index(match)

        if score > self.best_conf[matchIndex]:
            self.target_gps[matchIndex] = gps
            self.best_conf[matchIndex] = score


if __name__ == '__main__':
    ta = TargetAggregator('targets.csv')
    print(ta.targets)
