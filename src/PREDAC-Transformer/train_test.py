import torch
import numpy as np
import time
import sys
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import copy
from model import PredacTransformer 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
import torch.nn.functional as F

def train(train_dataset, val_dataset, args):
    device = args.device
    at_bank = args.at_bank
    sr_bank = args.sr_bank
    seq_len = at_bank.shape[1]
    feature_dim = at_bank.shape[2] * 2  
 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PredacTransformer(
        seq_len=seq_len, 
        feature_dim=feature_dim, 
        subtype=args.subtype, 
        dropout=0.1,
    ).to(device)
    
    best_model = model
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.CrossEntropyLoss()
    
    pb_miniters = 20
    pbar = tqdm(range(1, args.epochs + 1), miniters=pb_miniters)
    best_val_loss = 1e6

    for epoch in pbar:
        model.train()
        epoch_loss = 0
        
        for i, data in enumerate(train_loader):
    
            if len(data) == 3:
                at_idx, sr_idx, labels = data
            else:
                at_idx, sr_idx, labels = data[0], data[1], data[2]
            at_idx = at_idx.to(device)
            sr_idx = sr_idx.to(device)
            labels = labels.to(device)
            feat_at = at_bank[at_idx]  
            feat_sr = sr_bank[sr_idx]  
            images = torch.cat([feat_at, feat_sr], dim=2) 
            mid_point = images.shape[2] // 2
            images_aug = torch.roll(images, shifts=mid_point, dims=2)
            
            images_combined = torch.cat([images, images_aug], dim=0)
            labels_combined = torch.cat([labels, labels], dim=0)
           
            if labels_combined.dim() == 2 and labels_combined.shape[1] > 1:
                labels_combined = torch.argmax(labels_combined, dim=1)
            elif labels_combined.dim() == 2 and labels_combined.shape[1] == 1:
                labels_combined = labels_combined.squeeze()
            labels_combined = labels_combined.long()

            optimizer.zero_grad()
            outputs = model(images_combined)
            loss = criterion(outputs, labels_combined)
            loss.backward()

            clip_grad_value_(model.parameters(), clip_value=0.5)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(train_loader)
        pbar.set_description(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                if len(data) == 3:
                    val_at_idx, val_sr_idx, val_labels = data
                else:
                    val_at_idx, val_sr_idx, val_labels = data[0], data[1], data[2]
               
                val_at_idx = val_at_idx.to(device)
                val_sr_idx = val_sr_idx.to(device)
                val_labels = val_labels.to(device)
                val_feat_at = at_bank[val_at_idx]
                val_feat_sr = sr_bank[val_sr_idx]
                val_images = torch.cat([val_feat_at, val_feat_sr], dim=2)

                if val_labels.dim() == 2 and val_labels.shape[1] > 1:
                    val_labels = torch.argmax(val_labels, dim=1)
                elif val_labels.dim() == 2 and val_labels.shape[1] == 1:
                    val_labels = val_labels.squeeze()
                val_labels = val_labels.long() 
                
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            
    return best_model

def test(test_dataset, model, args):

    device = args.device
    
    at_bank = args.at_bank
    sr_bank = args.sr_bank
    
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    all_probs = []
    
    with torch.no_grad():
        for data in test_loader:
            if len(data) == 3:
                t_at_idx, t_sr_idx, _ = data 
            else:
                t_at_idx, t_sr_idx = data[0], data[1]
                
            t_at_idx = t_at_idx.to(device)
            t_sr_idx = t_sr_idx.to(device)
            t_feat_at = at_bank[t_at_idx]
            t_feat_sr = sr_bank[t_sr_idx]
            t_images = torch.cat([t_feat_at, t_feat_sr], dim=2)
            
            outputs = model(t_images)
            
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().detach().numpy())
            
    return np.array(all_probs).squeeze()

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    
    optied_model = train(train_ds, val_ds, args)
    
    return test(test_ds, optied_model, args)