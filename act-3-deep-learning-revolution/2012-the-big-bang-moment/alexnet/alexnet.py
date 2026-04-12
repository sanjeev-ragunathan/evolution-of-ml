'''
We're gonna be implementing AlexNet (2012) by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton.
For predicting images from the ImageNet dataset.

BUT - not on ImageNet - we'll be using a smaller dataset called CIFAR-10 (60k images, 10 classes) to train and test our AlexNet implementation.

The differences we make to adapt to the CIFAR-10:

'''

# ruff: noqa: E402 # to ignore "imports not on top of the file" warning

import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        # define layers
        self.conv1 = nn.Conv2d(3, 96, 3) # CIFR-10: RGB therefore 3 in_channel
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(96, 256, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5) # randomly turns off 50% of the neurons

        self.fc1 = nn.Linear(2304, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    
    def forward(self, image):
        # define data flow
        conv1_out = self.conv1(image)
        relu1_out = torch.relu(conv1_out)
        pool1_out = self.pool1(relu1_out)
        conv2_out = self.conv2(pool1_out)
        relu2_out = torch.relu(conv2_out)
        pool2_out = self.pool2(relu2_out)
        conv3_out = self.conv3(pool2_out)
        relu3_out = torch.relu(conv3_out)
        conv4_out = self.conv4(relu3_out)
        relu4_out = torch.relu(conv4_out)
        conv5_out = self.conv5(relu4_out)
        relu5_out = torch.relu(conv5_out)
        pool5_out = self.pool5(relu5_out)
        
        flatten_out = pool5_out.view(-1, 256*3*3)

        dropped = self.dropout(flatten_out)
        fc1_out = self.fc1(dropped)
        relufc1_out = torch.relu(fc1_out)
        dropped = self.dropout(relufc1_out)
        fc2_out = self.fc2(dropped)
        relufc2_out = torch.relu(fc2_out)
        fc3_out = self.fc3(relufc2_out)

        return fc3_out

# DATASET

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data Augmentation - to artificially increase the size of our dataset and help our model generalize better. We will apply random horizontal flip and random crop to the training images.

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

import torch.optim as optim
import time

if __name__ == "__main__":

    model = AlexNet()
    print(model)

    # LOSS FUNCTION - calculate error and backpropagates error
    loss_fn = nn.CrossEntropyLoss()
    # OPTIMIZER - updates weights
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # TRAIN for 5 epochs. Structure: forward - calculate error - backpropagate error - update weights
    for epoch in range(5):
        start = time.time()
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        elapsed = time.time() - start
        remaining = elapsed * (4 - epoch)
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}s, Remaining: ~{remaining:.0f}s") # this whole time thing will help us analyse how much time is remaining
    
    # TESTING
    total, correct = 0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"Test Accuracy: {correct/total:.4f}")