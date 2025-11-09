import os
import sys
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import *
from features import *
from train_test import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, default=os.cpu_count(), help="Number of threads for parallel computation")
parser.add_argument("--seq_path", type=str, default="demo/demo_HA1.csv", help="Path to HA1 sequence data", required=True)
parser.add_argument("--anti_path", type=str, default="demo/demo_HI.csv", help="Path to antigenicity data", required=True)
parser.add_argument("--subtype", type=str, choices=["H1", "H3", "H5"], help="subtype", required=True)
parser.add_argument("--label", type=str, default="NAD_distance", choices=["NAD_class", "NAD_distance", "AHD_class", "AHD_distance"], required=True)
args = parser.parse_args()

# ------- load dataset -------
at_seqs, sr_seqs, labels, at_year, sr_year, seqs, at_name, sr_name, at_ph_type, sr_ph_type = load_data(args)
adaboost_feature = AdaBoostFeature()
features = adaboost_feature.construct_feature(at_seqs, sr_seqs, at_name, sr_name, at_ph_type, sr_ph_type, args)
dataset = FluDataset(features, labels, at_name, sr_name)
sample_num = features.shape[0]

# ------- train and test -------
train_val_idx, test_idx, _, _ = train_test_split(range(sample_num), range(sample_num), test_size=0.2, random_state=0)
train_idx, val_idx, _, _ = train_test_split(train_val_idx, train_val_idx, test_size=0.25, random_state=1)
predict_ret = evaluate_func(dataset, train_idx, val_idx, test_idx, args)