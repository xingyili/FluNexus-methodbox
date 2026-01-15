import torch
import torch.nn as nn
import numpy as np

class FluAttnModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 256, 128, 64], dropout=0.0):
        super(FluAttnModel, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

    def set_scaler(self, mean, std):
        self.y_mean.fill_(mean)
        self.y_std.fill_(std)

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def predict_distance(self, x):
        with torch.no_grad():
            out_scaled = self.forward(x)
            return out_scaled * self.y_std + self.y_mean