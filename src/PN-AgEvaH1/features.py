import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

class PNAgEvaH1Feature:
    def __init__(self):
        with open("src/PN-AgEvaH1/data/aaindex_feature_H1N1_ding.txt", 'r') as f2:
            self.lines_file2 = f2.readlines()
        with open("src/PN-AgEvaH1/data/NIEK910102_matrix.csv", 'r') as f3:
            self.lines_file3 = f3.readlines()     

    '''
    Function: Extract corresponding row data
    '''
    def find_amino_acid_features(self,amino_acid, lines):
        for line in lines:
            data = line.strip().split('\t')
            if data[0] == amino_acid:
                return data[1:]
        return [0,0,0,0,0,0,0,0]

    '''
    Function to find values in the feature matrix
    '''
    def find_value(self, char1, char2, lines_file3):
        index1 = None
        index2 = None
        if char1 =='-' or char2 =='-':
            return 0
        
        for i, line in enumerate(lines_file3):
            if char1 == line[0]:
                index1 = i
            if char2 == line[0]:
                index2 = i

        if index1 is None or index2 is None:
            return 111

        aa = index1
        bb = index2
        index2 = max(aa, bb)
        index1 = min(aa, bb)
        line_index = index2
        line = lines_file3[line_index]
        elements = line.strip().split(',')
        ans = elements[index1 - 1]
        
        return ans
    
    '''
    Comparison function: counts different amino acids between two sequences
    '''
    def compare_sequences(self, seq1, seq2):
        diff_count = 0
        for char1, char2 in zip(seq1, seq2):
            if char1 != char2:
                diff_count += 1
        if diff_count >= 15:
            return 1
        else:
            return 0

    def process_pair(self, sequence1, sequence2):
        list1 = []

        # Check if sequences are similar enough, no feature extraction
        list1.append(self.compare_sequences(sequence1, sequence2))

        # extact features
        for i in range(len(sequence1)):
            feature1 = self.find_value(sequence1[i], sequence2[i], self.lines_file3)
            if sequence1[i] == sequence2[i]:
                word_i_1 = self.find_amino_acid_features(sequence1[i], self.lines_file2)
                list1.extend(word_i_1)
                list1.append(feature1)
                word_i_2 = self.find_amino_acid_features(sequence2[i], self.lines_file2)
                list1.extend(word_i_2)
                list1.append(feature1)
            else:
                feature1 = self.find_value(sequence1[i], sequence2[i], self.lines_file3)
                word_i_1 = self.find_amino_acid_features(sequence1[i], self.lines_file2)
                list1.extend(word_i_1)
                list1.append(feature1)
                word_i_2 = self.find_amino_acid_features(sequence2[i], self.lines_file2)
                list1.extend(word_i_2)
                list1.append(feature1)
        
        return list1

    def construct_feature(self, at_seqs, sr_seqs, at_name, sr_name, at_ph_type, sr_ph_type, args):
        seq_feature = Parallel(n_jobs=args.n_jobs)(
            delayed(self.process_pair)(at_seq, sr_seq) 
            for at_seq, sr_seq in zip(at_seqs, sr_seqs)
        )
        seq_feature = np.array(seq_feature).astype(float)
        return seq_feature