import os
import json
import torch
import numpy as np
from fluattn_dynamic import FluAttnMiner

class FluAttnFeature:
    def __init__(self):
        self.aa_to_idx = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
        self.json_path_1 = "src/FluAttn/data/prd/aaindex1_dicts.json"
        self.json_path_2 = "src/FluAttn/data/prd/aaindex2_dicts.json"

    def _seq_to_idx_array(self, seq):
        return np.array([self.aa_to_idx.get(ch, -1) for ch in seq], dtype=np.int32)

    def construct_feature(self, at_seqs, sr_seqs, args):
        miner = FluAttnMiner()
        miner.initialize(self.json_path_1, self.json_path_2, device=args.device)
 
        idx1 = np.vstack([self._seq_to_idx_array(s) for s in at_seqs])
        idx2 = np.vstack([self._seq_to_idx_array(s) for s in sr_seqs])

        raw_features = np.stack([idx1, idx2], axis=1)

        return raw_features