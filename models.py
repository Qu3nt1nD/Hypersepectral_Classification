import torch
import torch.nn as nn
import torch.nn.functional as F




class PixelClassifier3D(nn.Module):

    def __init__(self, nc:int):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 2, kernel_size=(3,3,7))
        self.conv2 = nn.Conv3d(2, 8, kernel_size=(3,3,3))
        self.maxPool = nn.MaxPool3d(kernel_size=(1,2,2))

        self.layer = nn.Linear(1*8*192, nc)
        self.softmax = nn.Softmax()

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        #x = self.maxPool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        #x = self.maxPool(x)
        x = self.dropout(x)

        x = self.layer(x.view(-1, 8*192))
        x = self.softmax(x)

        return x

class PixelClassifier2D(nn.Module):

    def __init__(self, nc:int, num_features:int):
        super().__init__()

        self.norm = nn.BatchNorm2d(num_features=num_features)

        self.conv1 = nn.Conv2d(num_features, 300, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(300, 400, kernel_size=(3,3), padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(2,2))

        self.layer1 = nn.Linear(5*5*400, 1000)
        self.layer2 = nn.Linear(1000, nc)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):

        x = self.norm(x)

        x = F.relu(self.conv1(x))
        #x = self.maxPool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        #x = self.maxPool(x)
        x = self.dropout(x)

        x = self.layer1(x.view(-1, 5*5*400))
        x = self.layer2(x)
        x = self.softmax(x)

        return x