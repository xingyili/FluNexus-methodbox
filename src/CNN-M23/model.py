from torch import nn
import torch.nn.functional as F
class CNN_M23(nn.Module):
    def __init__(self):
        super(CNN_M23, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=11, out_channels=256, kernel_size=(2, 7), stride=(1, 3), padding=(1, 3))
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.norm0 = nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75)
        nn.init.normal_(self.conv0.weight, std=0.01)
        nn.init.constant_(self.conv0.bias, 0)
        
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.norm1 = nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75)
        nn.init.normal_(self.conv1.weight, std=0.01)
        nn.init.constant_(self.conv1.bias, 0)
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.norm2 = nn.LocalResponseNorm(size=3, alpha=0.0001, beta=0.75)
        nn.init.normal_(self.conv2.weight, std=0.01)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.fc6 = nn.Linear(in_features=256 * 6, out_features=256)
        self.fc_last = nn.Linear(in_features=256, out_features=1)
        nn.init.normal_(self.fc6.weight, std=0.01)
        nn.init.constant_(self.fc6.bias, 0)
        nn.init.normal_(self.fc_last.weight, std=0.01)
        nn.init.constant_(self.fc_last.bias, 0)

        self.dropout6 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = self.norm0(x)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = self.fc_last(x)
        return x