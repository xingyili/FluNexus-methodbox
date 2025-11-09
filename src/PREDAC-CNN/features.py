import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class PREDACCNNFeature:
    def __init__(self, args):
        if args.subtype == "H1":
            self.aaindex = pd.read_csv("src/PREDAC-CNN/data/aaindex_feature_H1N1.txt", sep='\t', index_col=0)
        elif args.subtype == "H3":
            self.aaindex = pd.read_csv("src/PREDAC-CNN/data/aaindex_feature_H3N2.txt", sep='\t', index_col=0)
        self.aj_dic = self.aaindex.T.to_dict()
    
    def get_AAindex_features_np(self, at_seq, sr_seq):
        att_num = 7
        AAindex_features_np = np.zeros((att_num * 2, len(at_seq)))
        for i, key in enumerate(self.aaindex.keys()):
            for j in range(len(at_seq)):
                AAindex_features_np[i, j] = self.aaindex.loc[at_seq[j], key]
                AAindex_features_np[i + att_num, j] = self.aaindex.loc[sr_seq[j], key]

        return AAindex_features_np
    
    def construct_feature(self, at_seqs, sr_seqs, args):
        feature = Parallel(n_jobs=args.n_jobs)(delayed(self.get_AAindex_features_np)(at_seq, sr_seq) for at_seq, sr_seq in zip(at_seqs, sr_seqs))
        return np.array(feature)