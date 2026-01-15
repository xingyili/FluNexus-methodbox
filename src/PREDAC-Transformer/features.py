import os
import torch
import numpy as np
import esm

class PREDACFeature:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.aaindex_dict = {}

    def _load_aaindex(self, path):
        if not os.path.exists(path):
            return {}
        adict = {}
        try:
            with open(path, 'r') as f:
                header = next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if not parts: continue
                    aa = parts[0]
                    vals = [float(x) for x in parts[1:]]
                    adict[aa] = vals
        except Exception as e:
            print(f"Error reading AAIndex file: {e}")
        return adict

    def get_esm_embeddings_batch(self, seqs, model, alphabet, batch_converter, batch_size=64):
        model.eval()
        embeddings = []
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            batch_data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_reprs = results["representations"][6]
            for j, seq_len in enumerate([len(s) for s in batch_seqs]):
                feat = token_reprs[j, 1 : 1 + seq_len].cpu().numpy()
                embeddings.append(feat)
        return embeddings

    def _process_single_type_sequences(self, unique_seqs, model, alphabet, batch_converter, target_len, phys_dim):
        esm_feats = self.get_esm_embeddings_batch(unique_seqs, model, alphabet, batch_converter, batch_size=64)
        
        seq_to_feat = {}
        feat_dim = 320 + phys_dim
        for i, seq in enumerate(unique_seqs):
            esm = esm_feats[i]
            matrix = []
            
            for j, aa in enumerate(seq):
                if j >= target_len: break
                vec_phys = self.aaindex_dict.get(aa, [0] * phys_dim)
                matrix.append(np.concatenate([esm[j], vec_phys]).astype(np.float32))
            
            while len(matrix) < target_len:
                matrix.append(np.zeros(feat_dim, dtype=np.float32))
            
            seq_to_feat[seq] = np.array(matrix, dtype=np.float32)
            
        return seq_to_feat

    def construct_feature_dict(self, at_seqs, sr_seqs, args):

        subtype = args.subtype
        if subtype == "H3":
            target_len = 328
        else:
            target_len = 327
            
        aaindex_path = f"src/PREDAC-Transformer/data/aaindex/aaindex_feature_{subtype}.txt"
        self.aaindex_dict = self._load_aaindex(aaindex_path)
        sample_vals = next(iter(self.aaindex_dict.values()))
        phys_dim = len(sample_vals)

        unique_at_seqs = list(set(at_seqs))
        unique_sr_seqs = list(set(sr_seqs))
 
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        model.eval().to(self.device)
        batch_converter = alphabet.get_batch_converter()
  
        at_feat_dict = self._process_single_type_sequences(
            unique_at_seqs, model, alphabet, batch_converter, target_len, phys_dim
        )
        
        sr_feat_dict = self._process_single_type_sequences(
            unique_sr_seqs, model, alphabet, batch_converter, target_len, phys_dim
        )
        
        del model
        torch.cuda.empty_cache()
        
        return at_feat_dict, sr_feat_dict