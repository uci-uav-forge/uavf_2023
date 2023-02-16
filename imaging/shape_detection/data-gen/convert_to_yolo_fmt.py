import os
IMAGE_SIZE=512
OUTPUT_DIR_NAME="labels"
os.makedirs(OUTPUT_DIR_NAME, exist_ok=True)
for split in ["test", "train", "validation"]:
    os.makedirs(f"{OUTPUT_DIR_NAME}/{split}", exist_ok=True)
    file = f"{split}annotations.csv"
    with open(f"images/{file}", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            image,label,xmin,ymin,xmax,ymax = line.split(",")
            center_x = (int(xmin) + int(xmax)) / 2
            center_y = (int(ymin) + int(ymax)) / 2
            width  = int(xmax) - int(xmin)
            height = int(ymax) - int(ymin)
            # append to file if it doesn't exist, otherwise create it
            with open(f"{OUTPUT_DIR_NAME}/{split}/{image[:-4]}.txt", "a+") as f2:
                f2.write(f"{int(label)-1} {center_x/IMAGE_SIZE} {center_y/IMAGE_SIZE} {width/IMAGE_SIZE} {height/IMAGE_SIZE}\n")