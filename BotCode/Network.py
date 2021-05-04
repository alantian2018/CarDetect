import torch
import torch.nn as nn
import torchvision
from LoadData import *

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        hidden1 = nn.Sequential(
        nn.Conv2d(3, 10, kernel_size=5),
        nn.Conv2d(10, 15, kernel_size=5),
        nn.ReLU(),
        )
        self.feature = nn.Sequential(
            hidden1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=15*32*32, out_features=500),
            nn.Linear(500,250),
            nn.Linear(250,100),
            nn.Linear(in_features=100, out_features=74)
        )

    def forward(self, x):
        output = (self.feature(x))
   #     print (output.size())
        output1 = output.view(-1,15*32*32)
        output = (self.classifier(output1))

        return output



#train_loader = torch.utils.data.Dataloader()
