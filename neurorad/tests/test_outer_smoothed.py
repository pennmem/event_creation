from make_outer_surface import make_smoothed_surface
import os.path as osp
from config import paths
import argparse
from nibabel import freesurfer as nf
from scipy.spatial import distance as dist
import json

def test_make_smoothed_surface(subject, db_root):
    for hemi in ['lh','rh']:
        old_coords,new_coords = get_surfaces(subject,db_root,hemi=hemi)
        assert new_coords.shape == old_coords.shape
        assert (new_coords==old_coords).all()
        print("Success!")


def get_surfaces(subject,db_root,hemi):
    temp = subject.split('_')
    localization = '0' if len(temp)==1 else temp[1]
    subject = temp[0]


    surface_file = osp.join(paths.rhino_root,db_root,'protocols','r1','subjects',subject,'localizations',localization,'neuroradiology',
                              'current_source','surf','%s.pial'%hemi)
    outer_smoothed = make_smoothed_surface(surface_file)
    reference_smoothed_file = osp.join(paths.rhino_root,'data','eeg','freesurfer','subjects',subject,'surf','%s.pial-outer-smoothed'%hemi)
    print('reference_smoothed_file:\n%s'%reference_smoothed_file)
    new_coords,_ = nf.read_geometry(outer_smoothed)
    old_coords,_ = nf.read_geometry(reference_smoothed_file)

    return old_coords,new_coords


def compare_smoothed_surfaces(subject,db_root):
    dists = []
    means = []
    for hemi in ['lh','rh']:
        old_coords,new_coords = get_surfaces(subject,db_root,hemi=hemi)
        dists.append(max(dist.directed_hausdorff(old_coords,new_coords)[0],dist.directed_hausdorff(new_coords,old_coords)[0]))
        edist = dist.cdist(old_coords,new_coords).min(1).mean(0)
        print('%s:\t %s \t %s'%(hemi,dists[-1],edist))
        means.append(edist)
    return dists,means


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subjects',nargs='+')
    args = parser.parse_args()
    surf_dists = {}
    for subject in args.subjects:
        print('Testing %s'%subject)
        try:
            dists,means = compare_smoothed_surfaces(subject,'/home1/leond')
            surf_dists[subject]={'max':{'lh':dists[0],'rh':dists[1]},
                                 'mean':{'lh':means[0],'rh':means[1]}}
        except IOError:
            continue
    format_str = '%s\t%s\t%s'
    print(format_str%('Subject','lh','rh'))
    for subject in surf_dists:
        print(format_str%(subject,surf_dists[subject].get('lh'),surf_dists[subject].get('rh')))
    with open(osp.join(osp.dirname(__file__),'surface_dists.json'),'w') as results:
        json.dump(surf_dists,results,indent=2)