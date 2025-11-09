import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class CNNM23Feature:
    def __init__(self):
        AA_pca_factors = pd.read_csv("src/CNN-M23/AAindex_PCA_Factors.csv", delimiter=',', skiprows=0, header=0)
        self.AA_feature_dict = {key: np.array(value) for key, value in AA_pca_factors.to_dict('list').items()}

    def get_virus_features_np(self, HA):
        AAindex_features_np = np.zeros((11, 1, len(HA)))
        for i in range(11):
            for j in range(len(HA)):
                if HA[j] == "-":
                    AAindex_features_np[i][0][j] = 0
                else:
                    AAindex_features_np[i][0][j] = self.AA_feature_dict[HA[j]][i]
        return AAindex_features_np
    
    def process_sequence(self, at_seq, sr_seq):
        feature_seq1 = self.get_virus_features_np(at_seq)
        feature_seq2 = self.get_virus_features_np(sr_seq)
        return np.concatenate((feature_seq1, feature_seq2), axis=1)

    def construct_feature(self, at_seqs, sr_seqs, args):
        feature = Parallel(n_jobs=args.n_jobs)(delayed(self.process_sequence)(at_seq, sr_seq) for at_seq, sr_seq in zip(at_seqs, sr_seqs))
        feature = np.array(feature)
        return feature