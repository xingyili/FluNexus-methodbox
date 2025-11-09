import torch
import torch.nn as nn
import torch.nn.functional as F
class PREDAC_CNN(nn.Module):
    def __init__(self, seq_length):
        super(PREDAC_CNN, self).__init__()
        number_columns = 14
        number_filters = 6
        filter_size = 3
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=number_columns, out_channels=number_filters, kernel_size=filter_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=number_filters, out_channels=number_filters, kernel_size=filter_size, stride=1, padding='same')
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(number_filters * ((seq_length ) // 2) , 128) 
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x