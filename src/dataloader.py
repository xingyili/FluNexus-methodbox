import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch

class FluDataset(Dataset):
    def __init__(self, x, y, at_name=None, sr_name=None):
        self.x = x
        self.y = y
        self.at_name = at_name
        self.sr_name = sr_name

    def __len__(self): 
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]).float(), torch.tensor(self.y[idx])
    
    def subset(self, sub_idx):
        sub_at_name = None
        sub_sr_name = None
        if self.at_name is not None and self.sr_name is not None:
            sub_at_name = [self.at_name[i] for i in sub_idx]
            sub_sr_name = [self.sr_name[i] for i in sub_idx]
        return FluDataset(self.x[sub_idx], self.y[sub_idx],sub_at_name, sub_sr_name)

def load_data(args):
    seq_df = pd.read_csv(args.seq_path, sep=',')
    anti_df = pd.read_csv(args.anti_path, sep=',')

    if args.label == "NAD_distance":
        anti_df = anti_df[anti_df["NAD_distance"].notna()]
        labels = anti_df["NAD_distance"].to_numpy()
        at_ph_type = anti_df["NAD_at_ph"].tolist()
        sr_ph_type = anti_df["NAD_sr_ph"].tolist()
    elif args.label == "AHD_distance":
        anti_df = anti_df[anti_df["AHD_distance"].notna()]
        labels = anti_df["AHD_distance"].to_numpy()
        at_ph_type = anti_df["AHD_at_ph"].tolist()
        sr_ph_type = anti_df["AHD_sr_ph"].tolist()
    elif args.label == "NAD_class": 
        anti_df = anti_df[anti_df["NAD_class"].notna()]
        labels = anti_df["NAD_class"].to_numpy()
        at_ph_type = anti_df["NAD_at_ph"].tolist()
        sr_ph_type = anti_df["NAD_sr_ph"].tolist()
    elif args.label == "AHD_class":
        anti_df = anti_df[anti_df["AHD_class"].notna()]
        labels = anti_df["AHD_class"].to_numpy()
        at_ph_type = anti_df["AHD_at_ph"].tolist()
        sr_ph_type = anti_df["AHD_sr_ph"].tolist()

    seqs = seq_df['seq'].tolist()
    
    idx_to_seq = seq_df.set_index('index')['seq']
    idx_to_name = seq_df.set_index('index')['name']
    anti_df['at_seq'] = anti_df['at_index'].map(idx_to_seq)
    anti_df['sr_seq'] = anti_df['sr_index'].map(idx_to_seq)
    anti_df['at_name'] = anti_df['at_index'].map(idx_to_name)
    anti_df['sr_name'] = anti_df['sr_index'].map(idx_to_name)
    at_seqs = anti_df["at_seq"].tolist()
    sr_seqs = anti_df["sr_seq"].tolist()
    at_name = anti_df["at_name"].tolist()
    sr_name = anti_df["sr_name"].tolist()
    

    pair_year_large = np.maximum(anti_df['at_year'], anti_df['sr_year']).tolist()
    
    
    at_year = anti_df['at_index'].map(seq_df.set_index('index')['year']).tolist()
    sr_year = anti_df['sr_index'].map(seq_df.set_index('index')['year']).tolist()

    return at_seqs, sr_seqs, labels, at_year, sr_year, seqs, at_name, sr_name, at_ph_type, sr_ph_type