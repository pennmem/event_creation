from nibabel import nifti1
import numpy as np
from .config import Paths
import subprocess
import os.path as osp

def map_coords(coordinates,nifti_file):
    """
    Apply the RAS+ transform in a nifti file to a set of coordinates in voxel space
    :param coordinates: Array-like with shape (3,N) or (4,N)
    :param nifti_file:
    :return:
    """
    transform = nifti1.load(nifti_file).get_affine()

    coordinates = np.matrix(coordinates)
    if coordinates.shape[0]==3:
        coordinates = np.stack([coordinates,np.ones(coordinates.shape[1])],axis=0)

    assert coordinates.shape[0] ==4

    ras_coords = coordinates * transform.astype(np.mat)
    return ras_coords[:3,:]


def t1_mri_to_mni(t1_mri_coords, imaging_root, subject, name=''):
    """
    Uses the ANTS binaries identified in config.Paths to map T1 MRI coordinates into MNI space,
    via a set of transformations specified by files located at imaging_root
    :param t1_mri_coords {np.array}: An Nx3 array of MRI coordinates
    :param imaging_root: The path to the imaging directory
    :param subject:
    :param name:
    :return {np.array}: An Nx3 array of MNI coordinates
    """
    t1_mri_coords[:,:2] *= -1
    antsApplyTransform = osp.join(Paths.ants_root,'antsApplyTransformsToPoints')
    t1_to_itk = osp.join(imaging_root,'T01_CT_to_T00_mprageANTs0GenericAffine_RAS_itk.txt')
    template_to_subject_affine = osp.join(imaging_root,'T00','thickness',
                                          '{sub}TemplateToSubject0GenericAffine.mat'.format(sub=subject))
    template_to_subject_warp  = osp.join(imaging_root,'T00','thickness',
                                         '{sub}TemplateToSubject1InverseWarp.nii.gz'.format(sub=subject))
    faffine = osp.join(osp.dirname(Paths.ants_root),'localization','template_to_NickOasis','ch22t0GenericAffine.mat')
    finversewarp = osp.join(osp.dirname(faffine),'ch22t1InverseWarp.nii.gz')
    if name:
        name = name + '_'

    mri_filename = osp.join(imaging_root,'electrode_coordinates_%sT1.csv'%name)
    with open(mri_filename,'w') as mri_coords_file:
        mri_coords_file.write('x,y,z,t\n')
        for line in t1_mri_coords:
            mri_coords_file.write(','.join(['{:.4f}'.format(x) for x in line])+',0\n')
    tmp_file = osp.join(imaging_root,'tmp.csv')
    mni_filename = mri_filename.replace('T1','mni_from_T1')
    subprocess.check_call([antsApplyTransform,'-d','3','-i',mri_filename,'-o',tmp_file,'-t',t1_to_itk])
    args = [antsApplyTransform, '-d','3','-i',tmp_file,'-o',mni_filename,
                     '-t','[%s,1]'%t1_to_itk,
                    '-t', '[%s,1]'%template_to_subject_affine,
                    '-t', template_to_subject_warp,
                    '-t','[%s,1]'%faffine,
                    '-t', finversewarp]
    print('Calling: ', ' '.join(args))
    subprocess.check_call(args)
    corrected_mni_coordinates = np.loadtxt(mni_filename,delimiter=',',skiprows=1)
    corrected_mni_coordinates[:,:2] *= -1
    return corrected_mni_coordinates[:,:3]


def add_corrected_mni_cordinates(localization,imaging_root,subject):
    """
    Adds corrected mni coordinates to a localization. Requires that corrected T1 coordinates be present.
    :param localization:
    :param imaging_root:
    :param subject:
    :return:
    """
    corrected_t1_coordinates = localization.get_contact_coordinates('t1_mri',coordinate_type='corrected')
    corrected_mni_coordinates = t1_mri_to_mni(corrected_t1_coordinates,imaging_root,subject,'corrected')
    localization.set_contact_coordinates('mni',localization.get_contacts(),corrected_mni_coordinates,coordinate_type='corrected')