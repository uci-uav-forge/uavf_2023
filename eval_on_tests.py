from imaging.pipeline import Pipeline
import os
class MockLocalizer:
    def __init__(self, folder_name, index=-1):
        self.folder_name = folder_name
        if index==-1:
            self.index=0
            self.index_locked = False
        else:
            self.index = index
            self.index_locked = True

    def get_current_pos_and_angles(self):
        content = open(f"gopro_tests/{self.folder_name}/location{self.index}.txt").read()
        x,y,z,angles = content.split("\n")
        x, y, z = map(lambda s: float(s.split(" ")[1]), [x,y,z])
        angles = map(float, angles[1:-1].split(", "))
        if not self.index_locked:
            self.index += 1
        return tuple([x,y,z]), tuple(angles)
    
if __name__=="__main__":
    dir_name = "09-08|03_04_24"
    img_number = 57
    localizer = MockLocalizer(dir_name, index=57)
    imaging_pipeline = Pipeline(localizer, (5568, 4176), img_file=f"gopro_tests/{dir_name}/img{img_number}.png", targets_file='imaging/targets.csv')
    imaging_pipeline.run(num_loops=1)
