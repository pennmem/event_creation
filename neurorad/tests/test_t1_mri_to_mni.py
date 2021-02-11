from map_mni_coords import t1_mri_to_mni
import os.path as osp
from config import Paths
import numpy as np



def test_t1_mri_to_mni(subject):
    imaging_root = osp.join(Paths.rhino_root,'data','eeg',subject,'imaging',subject)
    t1_coord_file = osp.join(imaging_root,'electrode_coordinates_T1.csv')
    name = 'test'
    t1_coords =np.loadtxt(t1_coord_file,delimiter=',')[:,:3]
    new_mni_coords = t1_mri_to_mni(t1_coords,imaging_root,subject,name)
    old_mni_coords = np.loadtxt(osp.join(imaging_root,'electrode_coordinates_mni.csv'),delimiter=',')[:,:3]
    max_diff =  np.abs(new_mni_coords-old_mni_coords).max()
    rel_diff = max_diff/np.abs(old_mni_coords.mean())
    print('Largest discrepancy for subject ',subject,': ',max_diff)
    return max_diff,rel_diff


if __name__ =='__main__':
    with open('mni_diffs.csv','w') as mni_diffs:
        print('subject',',','Abs diff',',','Rel diff', file=mni_diffs)
        for subject in ['R1304N','R1286J','R1364C','R1254E','R1334T']:
            diffs = test_t1_mri_to_mni(subject)
            print(subject, ',', diffs[0],diffs[1], file=mni_diffs)

