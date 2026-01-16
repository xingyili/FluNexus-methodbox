# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import *
from features import *
from train_test import *
import argparse
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split 

class IndexBasedFluDataset(Dataset):
    def __init__(self, at_indices, sr_indices, labels):
        self.at_indices = at_indices
        self.sr_indices = sr_indices
        self.labels = labels
        self.y = labels 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.at_indices[idx], self.sr_indices[idx], self.labels[idx]
    
    def subset(self, indices):
        import numpy as np
        idx_arr = np.array(indices)
        at_sub = np.array(self.at_indices)[idx_arr].tolist()
        sr_sub = np.array(self.sr_indices)[idx_arr].tolist()
        if isinstance(self.labels, torch.Tensor):
            labels_sub = self.labels[idx_arr]
        elif isinstance(self.labels, np.ndarray):
            labels_sub = self.labels[idx_arr]
        else:
            labels_sub = np.array(self.labels)[idx_arr].tolist()
            
        return IndexBasedFluDataset(at_sub, sr_sub, labels_sub)
    
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--n_jobs", type=int, default=os.cpu_count())
# - -- - - 
parser.add_argument("--seq_path", type=str, default="demo/demo_HA1.csv", help="Path to HA1 sequence data", required=True)
parser.add_argument("--anti_path", type=str, default="demo/demo_HI.csv", help="Path to antigenicity data", required=True)
parser.add_argument("--subtype", type=str, choices=["H1", "H3", "H5"], help="subtype", required=True)
parser.add_argument("--label", type=str, default="NAD_class", choices=["NAD_class", "NAD_distance", "AHD_class", "AHD_distance"], required=True)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()
if args.device == "auto":
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

at_seqs, sr_seqs, labels, at_year, sr_year, seqs, at_name, sr_name, _, _ = load_data(args)

predac_transformer_feature = PREDACFeature()
at_feat_dict, sr_feat_dict = predac_transformer_feature.construct_feature_dict(at_seqs, sr_seqs, args)
unique_at_seqs = list(at_feat_dict.keys())
at_str2idx = {seq: i for i, seq in enumerate(unique_at_seqs)}
at_tensor_bank = torch.tensor(np.array([at_feat_dict[s] for s in unique_at_seqs]), dtype=torch.float32).to(args.device)
unique_sr_seqs = list(sr_feat_dict.keys())
sr_str2idx = {seq: i for i, seq in enumerate(unique_sr_seqs)}
sr_tensor_bank = torch.tensor(np.array([sr_feat_dict[s] for s in unique_sr_seqs]), dtype=torch.float32).to(args.device)
args.at_bank = at_tensor_bank
args.sr_bank = sr_tensor_bank
at_indices = [at_str2idx[s] for s in at_seqs]
sr_indices = [sr_str2idx[s] for s in sr_seqs]
dataset = IndexBasedFluDataset(at_indices, sr_indices, labels)
sample_num = len(dataset)

train_val_idx, test_idx, _, _ = train_test_split(
    range(sample_num), range(sample_num), test_size=0.2, random_state=0
)

train_idx, val_idx, _, _ = train_test_split(
    train_val_idx, train_val_idx, test_size=0.25, random_state=1
)

predict_ret = evaluate_func(dataset, train_idx, val_idx, test_idx, args)