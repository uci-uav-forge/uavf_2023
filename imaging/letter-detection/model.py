from imageDataset import *
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# change train dataset directory here
train_directory = './train/dataset'
test_directory = './test/dataset'
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = ImageDataset(train_directory, transform=transform)
test_dataset = ImageDataset(test_directory, transform=transform)
# loads the test dataset
train_loader = DataLoader(dataset = train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True)

num_epochs = 5

class CNN(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5))  # in_channels=1 bc input image is in grayscale, out_channels=32 for 32 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(kernel_size=(2,2))  # add stride as a parameter?
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5))
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=100352, out_features=64) # fully connected layer, like dense(64)
            
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 35)  # 35 is the number of classes
        self.sigmoid = nn.Sigmoid()
    
    # Progresses data across layers    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = F.relu(self.conv3(out))
        out = self.pool(out)
        out = self.flatten1(out)  
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


model = CNN()
# Set Loss function with criterion
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
# Set optimizer with optimizer
optimizer = torch.optim.Adam(model.parameters()) 

def train():
    # train the model
    model.train()
    print("Start training")
    for epoch in range(5):
        # loop = tqdm(train_loader, total=len(train_loader), leave=True)
        # for imgs, labels in loop:
        running_loss = 0.0
        count = 0
        for i, data in enumerate(test_loader, 0):
            imgs, labels = data
            # forward + backward + optimize
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
            """ loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = loss.item()) """
        print(f'Epoch {epoch+1}, Loss: {running_loss/count}')
    print("Finished training")
    torch.save(model, "./trained_model_torch")

def check_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    print('Accuracy of the network on the %d test images: %d %%' % (len(loader), 100 * correct / total))


#train()
model = torch.load('trained_model_torch')
check_accuracy(test_loader, model)

# Run to save model as onnx:  python -m tf2onnx.convert --saved-model trained_model --output model.onnx
