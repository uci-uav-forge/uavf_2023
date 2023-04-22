"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""
import cProfile
from imaging.pipeline import Pipeline
import time
import pstats
import io
from pstats import SortKey

class FakeLocalizer:
    def __init__(self):
        pass
    def get_current_location(self):
        return (69,-1337,420)
    def get_current_heading(self):
        return (-90,0,-90)

if __name__ == "__main__":
    imaging_pipeline = Pipeline(FakeLocalizer(), (5312, 2988), img_file="imaging/testimages/test1.png")
    #imaging_pipeline = Pipeline(FakeLocalizer(), (5312, 2988))
    start = time.perf_counter()

    #ob = cProfile.Profile()
    #ob.enable()
    imaging_pipeline.run(num_loops=1)

    #ob.disable()
    #sec = io.StringIO()
    #sortby = SortKey.CUMULATIVE
    #ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
    #ps.print_stats()

    #print(sec.getvalue())
    end = time.perf_counter()
    print(f"Time elapsed: {end - start}")
