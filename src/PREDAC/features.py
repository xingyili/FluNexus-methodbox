import numpy as np
import re
from Bio.PDB import PDBList, PDBParser
from joblib import Parallel, delayed
class PREDACFeature:
    def __init__(self):
        self.discretization_threshold_dict = {
            "epitope_A": 0, 
            "epitope_B": 2, 
            "epitope_C": 0, 
            "epitope_D": 1, 
            "epitope_E": 0, 
            "hydrophobicity": 1.82, 
            "volume": 54.67, 
            "charge": 2.49, 
            "accessible_surface_area": 34.87, 
            "polarity": 0.10, 
            "receptor_binding": 1.79, 
            "glycosylation": 1
        }
        self.epitopes_dict = {
            "epitope_A": np.array([i for i in range(140,147)]),
            "epitope_B": np.array([i for i in range(155,161)] + [i for i in range(187,197)]),
            "epitope_C": np.array([i for i in range(260,265)]),
            "epitope_D": np.array([i for i in range(275,281)]),
            "epitope_E": np.array([i for i in range(207,213)])
        }

        self.aa_pc_property_dict = {
            'hydrophobicity': {
                'A': -0.21, 'R': 2.11, 'N': 0.96, 'D': 1.36, 'C': -6.04, 'Q': 1.52, 'E': 2.30, 'G': 0.00, 'H': -1.23, 'I': -4.81, 
                'L': -4.68, 'K': 3.88, 'M': -3.66, 'F': -4.65, 'P': 0.75, 'S': 1.74, 'T': 0.78, 'W': -3.32, 'Y': -1.01, 'V': -3.50
            },
            'volume': {
                'A': 31.0, 'R': 124.0, 'N': 56.0, 'D': 54.0, 'C': 55.0, 'Q': 85.0, 'E': 83.0, 'G': 3.0, 'H': 96.0, 'I': 111.0, 
                'L': 111.0, 'K': 119.0, 'M': 105.0, 'F': 132.0, 'P': 32.5, 'S': 32.0, 'T': 61.0, 'W': 170.0, 'Y': 136.0, 'V': 84.0
            },
            'charge': {
                'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.05, 'Q': 5.65, 'E': 3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02, 
                'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.30, 'S': 5.68, 'T': 5.66, 'W': 5.89, 'Y': 5.66, 'V': 5.96
            },
            'polarity': {
                'A': 0.046, 'R': 0.291, 'N': 0.134, 'D': 0.105, 'C': 0.128, 'Q': 0.180, 'E': 0.151, 'G': 0.000, 'H': 0.230, 'I': 0.186, 
                'L': 0.186, 'K': 0.219, 'M': 0.221, 'F': 0.290, 'P': 0.131, 'S': 0.062, 'T': 0.108, 'W': 0.409, 'Y': 0.298, 'V': 0.140
            },
            'accessible_surface_area': {
                'A': 27.8, 'R': 94.7, 'N': 60.1, 'D': 60.6, 'C': 15.5, 'Q': 68.7, 'E': 68.2, 'G': 24.5, 'H': 50.7, 'I': 22.8, 
                'L': 27.6, 'K': 103.0, 'M': 33.5, 'F': 25.5, 'P': 51.5, 'S': 42.0, 'T': 45.0, 'W': 34.7, 'Y': 55.2, 'V': 23.7
            }
        }
        seq_coordinate = self.get_seq_coordinate('1HGF', 'A', 'CA')
        self.loop_130_dis, self.helix_190_dis, self.loop_220_dis = self.get_distance_residue2rb_region(seq_coordinate)


    def get_epitopes_diff_num(self, HA1, HA2, epitope_pos, discretization_threshold):
        HA1_np = np.array(list([HA1[i] for i in epitope_pos]))
        HA2_np = np.array(list([HA2[i] for i in epitope_pos]))
        return 1 if np.sum(HA1_np != HA2_np) > discretization_threshold else 0

    def get_pc_diff(self, HA1, HA2, aa_pc_property, discretization_threshold):

        HA1_pc_property_np = np.array([aa_pc_property[i] if i != '-' else 0 for i in HA1])
        HA2_pc_property_np = np.array([aa_pc_property[i] if i != '-' else 0 for i in HA2])
        diff_all_np = np.abs(HA1_pc_property_np - HA2_pc_property_np)
        diff_np = diff_all_np[diff_all_np != 0]
        sorted_diff_np = diff_np[np.argsort(diff_np)[::-1]]
        diff = np.mean(sorted_diff_np[:3]) if len(sorted_diff_np) > 3 else (0 if len(sorted_diff_np) == 0 else np.mean(sorted_diff_np))
        return 1 if diff > discretization_threshold else 0


    def get_ng_diff(self, HA1, HA2, discretization_threshold):
        pattern = r"N[AC-IK-NQ-TVWY][ST]"
        HA1_matches = re.finditer(pattern, HA1)
        HA2_matches = re.finditer(pattern, HA2)
        HA1_start_idices = [match.start() for match in HA1_matches]
        HA2_start_idices = [match.start() for match in HA2_matches]
        diff = 0
        diff = len(set(HA1_start_idices) | set(HA2_start_idices)) - len(set(HA1_start_idices) & set(HA2_start_idices))
        return 1 if diff > discretization_threshold else 0

    def get_seq_coordinate(self, pdb_id, chain_id, atom_id):
        pdb_parser = PDBParser(QUIET=True)
        HA1_structure = pdb_parser.get_structure(pdb_id, "src/PREDAC/pdb1hgf.ent")
        seq_coordinate = []
        for model in HA1_structure:
            for residue in model[chain_id]:
                if atom_id not in residue:
                    continue
                seq_coordinate.append(residue[atom_id].get_coord())
        return np.array(seq_coordinate)
 
    def get_distance_residue2rb_region(self, seq_coordinate):
        rb_region_idices = {
            'loop_130': [x for x in range(133,138)],
            'helix_190': [x for x in range(189,198)],
            'loop_220': [x for x in range(220,228)]
        }
        loop_130_dis = []
        helix_190_dis = []
        loop_220_dis = []
        for i in range(seq_coordinate.shape[0]):
            loop_130_dis.append(np.sqrt(np.sum((seq_coordinate[i] - seq_coordinate[rb_region_idices['loop_130']]) ** 2, axis=1)))
            helix_190_dis.append(np.sqrt(np.sum((seq_coordinate[i] - seq_coordinate[rb_region_idices['helix_190']]) ** 2, axis=1)))
            loop_220_dis.append(np.sqrt(np.sum((seq_coordinate[i] - seq_coordinate[rb_region_idices['loop_220']]) ** 2, axis=1)))
        return np.array(loop_130_dis), np.array(helix_190_dis), np.array(loop_220_dis)

    def get_rb_diff(self, HA1, HA2, discretization_threshold):
        diff_idx = np.where(np.array(list(HA1)) != np.array(list(HA2)))[0]
        diff_idx_distance = np.concatenate((self.loop_130_dis[diff_idx].reshape(-1), self.helix_190_dis[diff_idx].reshape(-1), self.loop_220_dis[diff_idx].reshape(-1)), axis=0)
        diff_idx_distance = np.sort(diff_idx_distance)
        diff = np.mean(diff_idx_distance[:3]) if len(diff_idx_distance) > 3 else (discretization_threshold + 1 if len(diff_idx_distance) == 0 else np.mean(diff_idx_distance))
        return 0 if diff > discretization_threshold else 1

    def compute_single_feature(self, at_seq, sr_seq):
        single_feature = []
        for key in ["epitope_A", "epitope_B", "epitope_C", "epitope_D", "epitope_E"]:
            single_feature.append(self.get_epitopes_diff_num(at_seq, sr_seq, self.epitopes_dict[key], self.discretization_threshold_dict[key]))
        for key in ["hydrophobicity", "volume", "charge", "polarity", "accessible_surface_area"]:
            single_feature.append(self.get_pc_diff(at_seq, sr_seq, self.aa_pc_property_dict[key], self.discretization_threshold_dict[key]))
        single_feature.append(self.get_ng_diff(at_seq, sr_seq, self.discretization_threshold_dict['glycosylation']))
        single_feature.append(self.get_rb_diff(at_seq, sr_seq, self.discretization_threshold_dict['receptor_binding']))
        return single_feature

    def construct_feature(self, at_seqs, sr_seqs, args):
        feature = Parallel(n_jobs=args.n_jobs)(delayed(self.compute_single_feature)(at_seq, sr_seq) for at_seq, sr_seq in zip(at_seqs, sr_seqs)) 
        return np.array(feature)