import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import copy
import os

class WeightedMultiHeadAttentionMLP(nn.Module):
    def __init__(self, seq_len, n_props, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        n_props = max(1, n_props) 
        
        self.attn_weights = nn.Parameter(torch.randn(n_heads, n_props)) 
        self.head_weights = nn.Parameter(torch.randn(n_heads))
        
        self.net = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        if x.shape[-1] != self.attn_weights.shape[-1]:
             if x.shape[-1] == 0:
                 raise RuntimeError(f"Input has 0 features but model expects {self.attn_weights.shape[-1]}. "
                                    "Check AAIndex file paths.")
        
        head_outputs = []
        for i in range(self.n_heads):
            attn = F.softmax(self.attn_weights[i], dim=-1)
            x_head = (x * attn.view(1, 1, -1)).sum(dim=-1)
            head_outputs.append(x_head)

        heads_stack = torch.stack(head_outputs, dim=1) 
        fusion_weights = F.softmax(self.head_weights, dim=0) 
        x_fused = torch.sum(heads_stack * fusion_weights.view(1, -1, 1), dim=1)
        return self.net(x_fused).squeeze(-1)

    def get_feature_importance(self, prop_names):
        with torch.no_grad():
            softmax_attn = F.softmax(self.attn_weights, dim=-1)
            fusion = F.softmax(self.head_weights, dim=0)
            final_weights = torch.sum(softmax_attn * fusion.view(-1, 1), dim=0)
            return final_weights.cpu().numpy()

class FluAttnMiner:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FluAttnMiner, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, json_path1, json_path2, device="cuda"):
        if self._initialized: return
        self.device = device
        self.aa_to_idx = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
        
        self.props1, self.names1 = self._load_aaindex1(json_path1)
        self.prop1_matrix = self._build_prop1_matrix(self.props1)

        self.props2, self.names2 = self._load_aaindex2(json_path2)
        
        if len(self.names1) == 0:
            raise FileNotFoundError(f"AAIndex1 file empty or not found: {json_path1}")
        if len(self.names2) == 0:
            raise FileNotFoundError(f"AAIndex2 file empty or not found: {json_path2}")

        self._initialized = True

    def _filter_correlated_props(self, raw_dict, threshold=0.8):
        prop_names = list(raw_dict.keys())
        aa_list = list("ARNDCQEGHILKMFPSTWYV")

        prop_matrix = np.array([
            [raw_dict[prop][aa] for aa in aa_list] for prop in prop_names
        ])
 
        corr_matrix = np.corrcoef(prop_matrix)
        keep = []
        removed = set()
        
        for i in range(len(prop_names)):
            if prop_names[i] in removed:
                continue
            keep.append(prop_names[i])
            for j in range(i + 1, len(prop_names)):
                if abs(corr_matrix[i, j]) > threshold:
                    removed.add(prop_names[j])
        return [raw_dict[k] for k in keep], keep

    def _load_aaindex1(self, path):
        if not os.path.exists(path): return [], []
        with open(path, 'r') as f: raw = json.load(f)

        filtered_props, filtered_names = self._filter_correlated_props(raw)
        
        processed = []
        for prop_dict in filtered_props:
            vals = np.array(list(prop_dict.values()))
            mean, std = vals.mean(), vals.std()
            if std == 0: std = 1.0
            processed.append({k: (v-mean)/std for k,v in prop_dict.items()})
            
        return processed, filtered_names

    def _build_prop1_matrix(self, props_list):
        if not props_list: return np.zeros((0, 20), dtype=np.float32)
        mat = np.zeros((len(props_list), 20), dtype=np.float32)
        for i, p in enumerate(props_list):
            for aa, idx in self.aa_to_idx.items():
                mat[i, idx] = p.get(aa, 0.0)
        return mat

    def _load_aaindex2(self, path):
        if not os.path.exists(path): 
            return [], []
        with open(path, 'r') as f: raw = json.load(f)
        names = list(raw.keys())
        matrices = []
        for name in names:
            mat = np.array(raw[name])
            min_val = mat.min()
            max_val = mat.max()
            if max_val > min_val:
                mat = mat / (max_val - min_val)
            else:
                mat = np.zeros_like(mat)
            matrices.append(mat)
        return matrices, names

    def mine_features(self, idx1_tr, idx2_tr, y_tr, idx1_val, idx2_val, y_val):
        y_mean, y_std = y_tr.mean(), y_tr.std()
        if y_std == 0: y_std = 1.0
        y_tr_z = (y_tr - y_mean) / y_std
        y_val_z = (y_val - y_mean) / y_std

        raw_X1_tr = self._get_tensor_prop1(idx1_tr, idx2_tr)
        raw_X1_val = self._get_tensor_prop1(idx1_val, idx2_val)
        res1 = self._run_optimization(raw_X1_tr, y_tr_z, raw_X1_val, y_val_z, 
                                      self.names1, n_retrain_heads=2)

        raw_X2_tr = self._get_tensor_prop2(idx1_tr, idx2_tr)
        raw_X2_val = self._get_tensor_prop2(idx1_val, idx2_val)
        res2 = self._run_optimization(raw_X2_tr, y_tr_z, raw_X2_val, y_val_z, 
                                      self.names2, n_retrain_heads=1)
        
        return res1, res2

    def construct_final_matrix(self, idx1, idx2, mining_results):
        (idx_p1, w_p1), (idx_p2, w_p2) = mining_results
        
        valid_mask = (idx1 >= 0) & (idx2 >= 0)
    
        X0 = ((idx1 != idx2) & valid_mask).astype(np.float32)
  
        raw_X1 = self._get_tensor_prop1(idx1, idx2, idx_p1) 
        X1 = np.dot(raw_X1, w_p1) 
        raw_X2 = self._get_tensor_prop2(idx1, idx2, idx_p2)
        X2 = np.dot(raw_X2, w_p2)
        
        return np.concatenate([X0, X1, X2], axis=1)

    def _run_optimization(self, X_tr, y_tr, X_val, y_val, names, n_retrain_heads, top_n=5):

        if X_tr.shape[-1] == 0:
            raise ValueError("No features available for mining! AAIndex loaded as empty.")

        model = WeightedMultiHeadAttentionMLP(X_tr.shape[1], X_tr.shape[2], n_heads=4).to(self.device)
        self._train_loop(model, X_tr, y_tr, X_val, y_val)
        
        imps = model.get_feature_importance(names)
        actual_top_n = min(len(names), top_n)
        if actual_top_n == 0:
             raise ValueError("Top-N selection resulted in 0 features.")

        top_indices = np.argsort(imps)[::-1][:actual_top_n].tolist()
        X_tr_sub = X_tr[:, :, top_indices]
        X_val_sub = X_val[:, :, top_indices]
        model_sub = WeightedMultiHeadAttentionMLP(X_tr.shape[1], len(top_indices), n_heads=n_retrain_heads).to(self.device)
        self._train_loop(model_sub, X_tr_sub, y_tr, X_val_sub, y_val)
        
        final_w = model_sub.get_feature_importance([names[i] for i in top_indices])
        return top_indices, final_w

    def _train_loop(self, model, X_tr, y_tr, X_val, y_val, epochs=200):
        if X_tr.shape[-1] == 0: return

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.MSELoss()
        
        xt = torch.from_numpy(X_tr).float()
        yt = torch.from_numpy(y_tr).float()
        xv = torch.from_numpy(X_val).float().to(self.device)
        yv = torch.from_numpy(y_val).float().to(self.device)
        
        dl = DataLoader(TensorDataset(xt, yt), batch_size=256, shuffle=True)
        
        best_loss = float('inf')
        patience, no_imp = 15, 0
        best_w = None
        
        for ep in range(epochs):
            model.train()
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
            
            model.eval()
            with torch.no_grad():
                v_loss = crit(model(xv), yv).item()
            
            if v_loss < best_loss:
                best_loss = v_loss
                best_w = copy.deepcopy(model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience: break
        
        if best_w: model.load_state_dict(best_w)

    def _get_tensor_prop1(self, idx1, idx2, subset_idx=None):
        valid = (idx1 >= 0) & (idx2 >= 0)
        mat = self.prop1_matrix
        if subset_idx is not None: mat = mat[subset_idx]
        
        if mat.shape[0] == 0:
            return np.zeros((idx1.shape[0], idx1.shape[1], 0), dtype=np.float32)

        p1 = mat[:, idx1] 
        p2 = mat[:, idx2]
        diff = np.abs(p1 - p2)
        diff[:, ~valid] = 0
        return diff.transpose(1, 2, 0) 
        
    def _get_tensor_prop2(self, idx1, idx2, subset_idx=None):
        N, L = idx1.shape
        props = self.props2
        if subset_idx is not None: props = [self.props2[i] for i in subset_idx]
        
        if len(props) == 0:
             return np.zeros((N, L, 0), dtype=np.float32)

        out = np.zeros((N, L, len(props)), dtype=np.float32)
        valid = (idx1 >= 0) & (idx2 >= 0)
        safe1 = np.where(valid, idx1, 0)
        safe2 = np.where(valid, idx2, 0)
        
        for i, mat in enumerate(props):
            vals = mat[safe1, safe2]
            vals[~valid] = 0
            out[:, :, i] = vals
        return out