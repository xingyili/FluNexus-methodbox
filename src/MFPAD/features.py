import numpy as np
from joblib import Parallel, delayed
import re

class MFPADFeature:
    def __init__(self):
        pass

    def feature_0_1(self, ha1, ha2):
        f=[]
        for i in range(len(ha1)):
            if ha1[i]==ha2[i]:
                f.append(0)
            elif ha1[i]!=ha2[i]:
                f.append(1)
        return f

    def mutation_number(self, ha1,ha2):
        mut_pos=[]
        for i in range(len(ha1)-1):
            if ha1[i] != ha2[i] and ha1[i]!="-" and ha2[i]!="-" and ha1[i]!="X" and ha2[i]!="X" :
                mut_pos.append(i)

        return mut_pos

    def im_site(self, ha1,ha2, subtype):
        if subtype == "H1":
            imsites=[124,126,127,128,139,141,142,152,153,154,155,156,157,186,190]
        elif subtype == "H3":
            imsites=[128,130,131,132,142,144,145,155,156,157,158,159,160,189,193]
        elif subtype == "H5":
            imsites=[123,125,126,127,138,140,141,151,152,153,154,155,156,185,189]
        is_mu=0
        for i in imsites:
            if ha1[i]!=ha2[i] and ha1[i]!="-" and ha2[i]!="-"and ha1[i]!="X" and ha2[i]!="X":
                is_mu=1
                break
        return is_mu

    def epitopes(self, ha1,ha2):
        is_mu=0
        for i in range(len(ha1)):
            if ha1[i]!=ha2[i] and ha1[i]!="-" and ha2[i]!="-"and ha1[i]!="X" and ha2[i]!="X":
                is_mu=1
                break
        return is_mu

    def gly_site(self, ha1,ha2):
        is_mut=0
        pos1=self.find_all(ha1)
        pos2=self.find_all(ha2)
        pos=list(set(pos1+pos2))
        for i in pos:
            if ha1[i] != ha2[i] or ha1[i+1] != ha2[i+1] or ha1[i+2] != ha2[i+2]:
                is_mut=1
                return is_mut
        return is_mut

    def phy_pro(self, ha1,ha2,mut_pos):
        phy={'A':[1.8,67,6,41],'R':[-4.5,148,10.76,-14],'N':[-3.5,91,5.41,-28],
            'D':[-3.5,109,2.77,-55],'C':[2.5,86,5.07,49],'Q':[-3.5,90,5.56,-10],
            'E':[-3.5,114,3.22,-31],'G':[-0.4,48,5.97,0],'H':[-3.2,118,7.59,8],
            'I':[4.5,124,6.02,99],'L':[3.8,124,5.98,97],'K':[-3.9,135,9.74,-23],
            'M':[1.9,124,5.74,74],'F':[2.8,135,5.48,100],'P':[-1.6,96,6.3,-46],
            'S':[-0.8,73,5.68,-5],'T':[-0.7,93,5.6,13],'W':[-0.9,163,5.89,97],
            'Y':[-1.3,141,5.66,63],'V':[4.2,105,5.96,76]}
        hyd_list = []
        vol_list = []
        char_list = []
        pol_list = []
        for i in mut_pos:
            h = abs(phy[ha1[i]][0] - phy[ha2[i]][0])
            v = abs(phy[ha1[i]][1] - phy[ha2[i]][1])
            c = abs(phy[ha1[i]][2] - phy[ha2[i]][2])
            p = abs(phy[ha1[i]][3] - phy[ha2[i]][3])
            hyd_list.append(h)
            vol_list.append(v)
            char_list.append(c)
            pol_list.append(p)
        hyd_list.sort(reverse=True)
        vol_list.sort(reverse=True)
        char_list.sort(reverse=True)
        pol_list.sort(reverse=True)
        if len(mut_pos)>3:
            hyd = (hyd_list[0]+hyd_list[1]+hyd_list[2])/3
            vol = (vol_list[0] + vol_list[1] + vol_list[2])/3
            char = (char_list[0] + char_list[1] + char_list[2])/3
            pol = (pol_list[0] + pol_list[1] + pol_list[2])/3
        elif len(mut_pos)>0:
            hyd = sum(hyd_list)/len(mut_pos)
            vol = sum(vol_list)/len(mut_pos)
            char = sum(char_list)/len(mut_pos)
            pol = sum(pol_list)/len(mut_pos)
        else :
            hyd = 0
            vol = 0
            char = 0
            pol = 0

        final_phy=[]
        final_phy.append(hyd)
        final_phy.append(vol)
        final_phy.append(char)
        final_phy.append(pol)
        return final_phy

    def find_all(self, ha):
        start=0
        gly='N.(T|S)'
        pos=[]
        for m in re.finditer(gly, ha):
            if ha[m.start()+1] != "P":
                pos.append(m.start())

        return pos
    def get_features(self, at_seq, sr_seq, subtype):
        fea=[]
        f=self.feature_0_1(at_seq,sr_seq)
        fea.extend(f)
        mut_pos=self.mutation_number(at_seq,sr_seq)
        f1_mutnum=len(mut_pos)
        fea.append(f1_mutnum)
        f2_imsite=self.im_site(at_seq,sr_seq, subtype)
        fea.append(f2_imsite)
        if subtype == "H1":
            f3_A = self.epitopes(at_seq[137:144], sr_seq[137:144])
            f4_B = self.epitopes(at_seq[152:158] + at_seq[184:194], sr_seq[152:158] + sr_seq[184:194])
            f5_C = self.epitopes(at_seq[257:263], sr_seq[257:263])  #E
            f6_D = self.epitopes(at_seq[273:279], sr_seq[273:279])  #C
            f7_E = self.epitopes(at_seq[204:210], sr_seq[204:210])  #D
        elif subtype == "H3":
            f3_A = self.epitopes(at_seq[140:147],sr_seq[140:147])
            f4_B = self.epitopes(at_seq[155:161] + at_seq[187:197], sr_seq[155:161] + sr_seq[187:197])
            f5_C = self.epitopes(at_seq[260:265], sr_seq[260:265])  #E
            f6_D = self.epitopes(at_seq[275:281], sr_seq[275:281])  #C
            f7_E = self.epitopes(at_seq[207:213], sr_seq[207:213])  #D
        elif subtype == "H5":
            f3_A = self.epitopes(at_seq[136:143],sr_seq[136:143])
            f4_B = self.epitopes(at_seq[151:157] + at_seq[183:193], sr_seq[151:157] + sr_seq[183:193])
            f5_C = self.epitopes(at_seq[256:262], sr_seq[256:262])  #E
            f6_D = self.epitopes(at_seq[272:278], sr_seq[272:278])  #C
            f7_E = self.epitopes(at_seq[203:209], sr_seq[203:209])  #D
        fea.append(f3_A)
        fea.append(f4_B)
        fea.append(f5_C)
        fea.append(f6_D)
        fea.append(f7_E)

        f8_gly = self.gly_site(at_seq,sr_seq)
        fea.append(f8_gly)

        f9_12=self.phy_pro(at_seq,sr_seq,mut_pos)
        fea=fea+f9_12
        return fea


    def construct_feature(self, at_seqs, sr_seqs, args):
        feature = Parallel(n_jobs=args.n_jobs)(
            delayed(self.get_features)(at_seq, sr_seq, args.subtype)
            for at_seq, sr_seq in zip(at_seqs, sr_seqs)
        )
        feature = np.array(feature)
        return feature