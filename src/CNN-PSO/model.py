import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_PSO(nn.Module):
    def __init__(self):
        super(CNN_PSO, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=193, kernel_size=(96, 1), stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(in_channels=193, out_channels=212, kernel_size=(5, 1), stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.dropout2 = nn.Dropout(p=0.165)

        self.conv3 = nn.Conv2d(in_channels=212, out_channels=109, kernel_size=(5, 1), stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.dropout3 = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(in_features=109 * 11, out_features=256)
        self.dropout4 = nn.Dropout(p=0.241)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return torch.sigmoid(x)