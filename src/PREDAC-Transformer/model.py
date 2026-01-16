import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class PredacTransformer(nn.Module):
    def __init__(self, seq_len, feature_dim, subtype="H3N2", dropout=0.1):
        super(PredacTransformer, self).__init__()

        self.d_model = 16 
        dim_feedforward = 64
        num_layers = 2
   
        if "H1" in subtype:
            num_heads = 4
        else:
            num_heads = 8

        self.pre_proj = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Linear(32, 16)
        )

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=1000)
        self.dropout_layer = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.flatten_dim = seq_len * self.d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(128, 2)    
        )

    def forward(self, x):

        x = self.pre_proj(x)
        x = self.pos_encoder(x)
        x = self.dropout_layer(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)