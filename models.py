import torch
import torch.nn as nn
import torch.nn.functional as F




class PixelClassifier(nn.Module):

    def __init__(self, nc:int):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 2, kernel_size=(3,3,7))
        self.conv2 = nn.Conv3d(2, 8, kernel_size=(3,3,3))
        self.maxPool = nn.MaxPool3d(kernel_size=(1,2,2))

        self.layer = nn.Linear(8*192, nc)
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