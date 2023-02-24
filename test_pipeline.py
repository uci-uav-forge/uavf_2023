"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""

from imaging.src.yolo_main import Pipeline
import time

class FakeLocalizer:
    def __init__(self):
        pass
    def get_current_location(self):
        return (69,420)

if __name__ == "__main__":
    imaging_pipeline = Pipeline(FakeLocalizer(), cam_mode="image")
    start = time.perf_counter()
    imaging_pipeline.run(num_loops=1)
    end = time.perf_counter()
    print(f"Time elapsed: {end - start}")
