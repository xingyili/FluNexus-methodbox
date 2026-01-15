import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class IVACNNFeature:
    def __init__(self, args):
        df = pd.read_csv('src/IAV-CNN/data/protVec_100d_3grams.csv', delimiter='\t')
        trigrams = list(df['words'])
        self.trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
        self.trigram_vecs = df.loc[:, df.columns != 'words'].values

    def get_embedding(self, at_seq, sr_seq):
        strain_embedding = []

        # Sequence length minus 2, 3 amino acids embedded in a 100-dimensional vector, each seq represented by a matrix of (seq_len-2) × 100
        for j in range(0, len(at_seq) - 2):
            at_trigram = at_seq[j:j + 3]
            # If three amino acids are encoded by three -
            # if at_trigram[0] == '-' or at_trigram[1] == '-' or at_trigram[2] == '-':
            if "-" in at_trigram:
                at_tri_embedding = self.trigram_vecs[self.trigram_to_idx['<unk>']]
            # command condition
            else:
                at_tri_embedding = self.trigram_vecs[self.trigram_to_idx[at_trigram]]

            sr_trigram = sr_seq[j:j + 3]
            # if trigram2[0] == '-' or trigram2[1] == '-' or trigram2[2] == '-':
            if '-' in sr_trigram:
                sr_tri_embedding = self.trigram_vecs[self.trigram_to_idx['<unk>']]
            else:
                sr_tri_embedding = self.trigram_vecs[self.trigram_to_idx[sr_trigram]]

            # It is straightforward to get a vector of differences between the three amino acids in the two seq's
            tri_embedding = at_tri_embedding - sr_tri_embedding
            strain_embedding.append(tri_embedding)
        return strain_embedding
        
    def construct_feature(self, at_seqs, sr_seqs, args):
        feature = Parallel(n_jobs=args.n_jobs)(delayed(self.get_embedding)(at_seq, sr_seq) for at_seq, sr_seq in zip(at_seqs, sr_seqs))
        feature = np.array(feature)
        return feature.reshape(feature.shape[0], 1, feature.shape[1], feature.shape[2])