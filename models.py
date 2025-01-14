import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(mode:str, dataset, num_features:int):

    if mode == "3D":
        model = PixelClassifier3D(nc=len(dataset.labels_list), num_features=num_features)
    elif mode == "2D":
        model = PixelClassifier2D(nc=len(dataset.labels_list), num_features=num_features)

    return model

class PixelClassifier3D(nn.Module):

    def __init__(self, nc:int, num_features:int):
        super().__init__()

        self.nf = num_features

        self.conv1 = nn.Conv3d(1, 2, kernel_size=(3,3,7))
        self.conv2 = nn.Conv3d(2, 8, kernel_size=(3,3,3))
        self.maxPool = nn.MaxPool3d(kernel_size=(1,2,2))

        self.layer = nn.Linear(1*8*(self.nf-8), nc)
        self.softmax = nn.Softmax()

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        #x = self.maxPool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        #x = self.maxPool(x)
        x = self.dropout(x)

        x = self.layer(x.view(-1, 8*(self.nf-8)))
        x = self.softmax(x)

        return x

class PixelClassifier2D(nn.Module):

    def __init__(self, nc:int, num_features:int):
        super().__init__()

        self.nf = num_features

        self.norm = nn.BatchNorm2d(num_features=self.nf)

        self.conv1 = nn.Conv2d(self.nf, self.nf, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(self.nf, self.nf, kernel_size=(3,3), padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(2,2))

        self.layer1 = nn.Linear(5*5*self.nf, 1000)
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

        x = self.layer1(x.view(-1, 5*5*self.nf))
        x = self.layer2(x)
        x = self.softmax(x)

        return x