import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np
from tqdm import tqdm  
from model import FluAttnModel 
from fluattn_dynamic import FluAttnMiner

def train_mlp_model(X_train, y_train, X_val, y_val, args):
    device = args.device
    y_mean, y_std = y_train.mean(), y_train.std()
    if y_std == 0: y_std = 1.0
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = FluAttnModel(input_dim=X_train.shape[1]).to(device)
    model.set_scaler(y_mean, y_std) 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    best_rmse = float("inf")
    best_model = None
    no_imp = 0
    pb_miniters = 20
    pbar = tqdm(range(1, args.epochs + 1), miniters=pb_miniters)
    
    for epoch in pbar:
        model.train()
        train_loss_sum = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            y_scaled = (y - y_mean) / y_std
            optimizer.zero_grad()
            loss = criterion(model(x), y_scaled)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            
        avg_train_loss = train_loss_sum / len(train_ds)
        pbar.set_description(f'Epoch {epoch}, Loss: {avg_train_loss:.4f}')

        model.eval()
        mse_sum = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model.predict_distance(x)
                mse_sum += ((pred - y) ** 2).sum().item()
        
        val_rmse = np.sqrt(mse_sum / len(val_ds))
        scheduler.step(val_rmse)
        pbar.set_postfix({
            'Loss': f'{avg_train_loss:.4f}', 
            'Val_RMSE': f'{val_rmse:.4f}', 
            'Best': f'{best_rmse:.4f}'
        })
        
        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_model = copy.deepcopy(model)
            
    return best_model

def predict(model, X, args):
    device = args.device
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x,) in dl:
            x = x.to(device)
            p = model.predict_distance(x)
            preds.extend(p.cpu().numpy())
    return np.array(preds)

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    raw_X = dataset.x
    raw_Y = dataset.y

    idx1_tr, idx2_tr = raw_X[train_idx, 0], raw_X[train_idx, 1]
    y_tr = raw_Y[train_idx]
    
    idx1_val, idx2_val = raw_X[val_idx, 0], raw_X[val_idx, 1]
    y_val = raw_Y[val_idx]
    
    idx1_test, idx2_test = raw_X[test_idx, 0], raw_X[test_idx, 1]
    miner = FluAttnMiner() 
    mining_results = miner.mine_features(idx1_tr, idx2_tr, y_tr, idx1_val, idx2_val, y_val)
    
    X_train = miner.construct_final_matrix(idx1_tr, idx2_tr, mining_results)
    X_val   = miner.construct_final_matrix(idx1_val, idx2_val, mining_results)
    X_test  = miner.construct_final_matrix(idx1_test, idx2_test, mining_results)
    
    model = train_mlp_model(X_train, y_tr, X_val, y_val, args)
    y_pred = predict(model, X_test, args)
    
    return y_pred