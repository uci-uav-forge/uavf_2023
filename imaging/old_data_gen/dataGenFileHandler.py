import os

def folder_empty(directory):
    return not (len(os.listdir(directory)) > 0)

def format_directory(dataset_dir: str):
    image_dir = dataset_dir + '/images'
    label_dir = dataset_dir + '/labels'
    full_label_dir = dataset_dir + '/full_labels'
    #make directories for dataset
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(full_label_dir, exist_ok=True)

def get_shapes(shape_dir):
    return sorted(i.split('.')[0] for i in os.listdir(shape_dir) if i.endswith('.obj') )
def get_alphas(alpha_dir):
    return sorted(i.split('.')[0] for i in os.listdir(alpha_dir) if i.endswith('.obj') )

def get_backgrounds(backgnd_dir):
    return [b.path for b in os.scandir(path=backgnd_dir) if b.is_file()]
