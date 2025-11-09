import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class CNNPSOFeature:
    def __init__(self):
        self.aa_idx = "ARNDCQEGHILKMFPSTWYV"
        self.AAindex_id = [
            "BENS940104",
            "LUTR910108",
            "MUET010101",
            "KOLA920101",
            "AZAE970101",
            "BONM030104",
            "TANS760101",
            "ZHAC000106",
            "BETM990101",
            "BONM030103",
        ]
        self.AAindex_values_dict = {}
        for id in self.AAindex_id:
            self.AAindex_values_dict[id] = self.get_AAindex_value_df(id)

    def get_AAindex_value_df(self, id):
        with open("src/CNN-PSO/AAindex/" + id + ".txt", "r") as file:
            lines = file.readlines()
            AAindex_value_matrix = []
            for line in lines:
                row = list(map(float, line.split()))
                AAindex_value_matrix.append(row + [np.nan] * (len(lines) - len(row)))
            AAindex_value_df = pd.DataFrame(AAindex_value_matrix)
            AAindex_value_df = AAindex_value_df.where(~np.triu(np.ones(AAindex_value_df.shape), 1).astype(bool), AAindex_value_df.T)
            AAindex_value_df.index = list(self.aa_idx)
            AAindex_value_df.columns = list(self.aa_idx)
            
            AAindex_value_df = (AAindex_value_df - AAindex_value_df.min().min()) / (AAindex_value_df.max().max() - AAindex_value_df.min().min())
            return AAindex_value_df

    def get_AAindex_features_np(self, HA1, HA2):
        AAindex_features_list = [[] for _ in range(len(self.AAindex_id))]
        for i in range(len(HA1)):
            for j in range(len(self.AAindex_id)):
                if (HA1[i] == "-" or HA2[i] == "-"):
                    AAindex_features_list[j].append(0)
                else:
                    AAindex_features_list[j].append(self.AAindex_values_dict[self.AAindex_id[j]][HA1[i]][HA2[i]])
        return np.array(AAindex_features_list).T

    def construct_feature(self, at_seqs, sr_seqs, args):
        feature = Parallel(n_jobs=args.n_jobs)(delayed(self.get_AAindex_features_np)(at_seq, sr_seq) for at_seq, sr_seq in zip(at_seqs, sr_seqs))
        feature = np.array(feature)
        return feature.reshape(feature.shape[0], 1, feature.shape[1], feature.shape[2])