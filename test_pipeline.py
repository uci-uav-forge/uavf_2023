#!/usr/bin/python3
"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""

from imaging.pipeline import Pipeline
import time

class FakeLocalizer:
    def __init__(self):
        pass
    def get_current_location(self):
        return (69,-1337,420)
    def get_current_heading(self):
        return (-90,0,-90)

if __name__ == "__main__":
    USE_GOPRO = False
    imaging_pipeline = Pipeline(FakeLocalizer(), (5312, 2988), img_file="gopro" if USE_GOPRO else "imaging/gopro-image-5k.png", targets = [("X", "trapezoid"), ("B", "square"), ("E", "circle")])
    start = time.perf_counter()
    imaging_pipeline.run(num_loops=1)
    end = time.perf_counter()
    print(f"Time elapsed: {end - start}")
