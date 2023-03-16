from imageDataset import *
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.onnx

from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# change train dataset directory here
train_directory = './train/dataset/'
test_directory = './test/dataset/'
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = ImageDataset(train_directory, transform=transform)
test_dataset = ImageDataset(test_directory, transform=transform)
# loads the test dataset
train_loader = DataLoader(dataset = train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True)

num_epochs = 50

class CNN(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5))  # in_channels=1 bc input image is in grayscale, out_channels=32 for 32 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(kernel_size=(5,5), stride=2)  # add stride as a parameter?
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5))
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=80000, out_features=64) # fully connected layer, like dense(64)
        #self.fc1 = nn.Linear(in_features=6272, out_features=64)    # size 32 images
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
    for epoch in range(num_epochs):
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
    torch.save(model, "./trained_model_torch_xx")

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
#check_accuracy(test_loader, model)

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 
    # create a dummy input tensor  
    #dummy_input = torch.randn((1, 5), input_size, requires_grad=True)  
    dummy_input = torch.randn(1, 1, 128, 128)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "LetterModel.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=13,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Converted to ONNX')

Convert_ONNX()