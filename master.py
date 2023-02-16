from imaging.yolo_main import Pipeline
from navigation.guided_mission.run_mission import Localizer

if __name__ == "__main__":
    localizer = Localizer()
    imaging_pipeline = Pipeline(localizer)
    imaging_pipeline.run()
