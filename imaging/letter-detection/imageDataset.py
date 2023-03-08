import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.root_dir = directory
        # change file name here
        self.annotations = pd.read_csv(directory + '/labels.txt')
        self.files = self.annotations['file'].values
        self.transform = transform
        """ self.labels = train_labels = df[' label'].values
        self.train_filepath_tensor = torch.from_numpy(files)
        self.train_label_tensor = torch.from_numpy(labels) """

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        img = Image.open(self.root_dir+img_path).convert("L")  # "L" means grayscale
        y_label = torch.tensor(self.annotations.iloc[index, 1])

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)