"""
converts electrode (or for that matter, any point) coordinates in MRI space
to coordinates in freesurfer mesh space.
Requires subject ID to find coords data (in electrodenames_coordinates_native_and_T1.csv) and
the two matrices Norig and Torig which can be obtained using mri_info


Run:
    python voxcoords_to_fs.py <subject> <out_file>
to create a voxel_coordinates_fs.json file
"""
import os.path as osp
import logging
from .mri_info import *
from numpy.linalg import inv
from .config import paths
from .localization import InvalidContactException
from nibabel.freesurfer import read_geometry,read_annot
from scipy.spatial import distance as dist
from functools import reduce
logger = logging.getLogger('submission')


def xdot(*args):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """
    return reduce(np.dot, args)


def read_and_tx(t1_file, fs_orig_t1,talxfmfile, localization,):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """

    # Get freesurfer matrices
    Torig = get_transform(fs_orig_t1, 'vox2ras-tkr')
    Norig = get_transform(fs_orig_t1, 'vox2ras')
    with open(talxfmfile) as txf:
        talxfm = np.matrix([x.strip().strip(';').split() for x in txf.readlines()[-3:]]).astype(float)
    logger.debug("Got transform")
    for line in open(t1_file):
        split_line = line.strip().split(',')

        # Contact name
        contact_name = split_line[0]

        # Contact location
        x = split_line[10]
        y = split_line[11]
        z = split_line[12]

        # Create homogeneous coordinate vector
        # coords = float(np.vectorize(np.matrix([x, y, z, 1.0])))
        coords = np.matrix([float(x), float(y), float(z), 1]).T
        
        # Compute the transformation
        fullmat = Torig * inv(Norig)
        fscoords = fullmat * coords
        tal_coords = talxfm * coords
        logger.debug("Transforming {}".format(contact_name))

        fsx = fscoords[0]
        fsy = fscoords[1]
        fsz = fscoords[2]

        fsavgx = tal_coords[0]
        fsavgy = tal_coords[1]
        fsavgz = tal_coords[2]

        # Enter into "leads" dictionary
        try:
            localization.set_contact_coordinate('fs', contact_name, [fsx, fsy, fsz], 'raw')
            localization.set_contact_coordinate('t1_mri', contact_name, [x, y, z])
            localization.set_contact_coordinate('tal',contact_name,[fsavgx,fsavgy,fsavgz],'raw')
        except InvalidContactException:
            logger.warn('Invalid contact %s in file %s'%(contact_name,os.path.basename(t1_file)))
    logger.debug("Done with transform")
    return Torig,Norig,talxfm

def map_to_average_brain(coords,left_pial,right_pial,left_sphere,right_sphere):
    """
    Maps a set of Freesurfer surface coordinates in an individual brain to the equivalent coordinates on the average
    brain.

    Method taken from the iElvis project (http://ielvis.pbworks.com), which implemented the same function in MATLAB:
    https://github.com/iELVis/iELVis/blob/master/iELVis_MAIN/iELVis_MATLAB/ELEC_LOC/sub2AvgBrain.m

    :param coords: {np.ndarray} Coordinates in the individual Freesurfer space
    :param subject_surf_dir: {str} Path to the directory containing the subject's surface meshes
    :param avg_surf_dir: {str} Path to the directory containing the fsaverage surface meshes
    :return: {np.ndarray} The matching coordinates in the average brain
    :return: {np.ndarray} The corresponding atlas labels in the average brain
    """
    hemispheres = ['left','right'] # For all surfaces, we append the right hemisphere to the left hemisphere
    fsavg_subj_dir  = osp.join(paths.rhino_root,'data','eeg','freesurfer','subjects','fsaverage',)
    files = {
        'left_pial':left_pial,
        'right_pial':right_pial,
        'left_sphere':left_sphere,
        'right_sphere':right_sphere,
        'left_avg_sphere':osp.join(fsavg_subj_dir,'surf','lh.sphere.reg'),
        'right_avg_sphere': osp.join(fsavg_subj_dir,'surf','rh.sphere.reg'),
        'left_avg_pial': osp.join(fsavg_subj_dir,'surf','lh.pial'),
        'right_avg_pial': osp.join(fsavg_subj_dir,'surf','rh.pial'),
        'left_avg_annot':osp.join(fsavg_subj_dir, 'label','lh.aparc.annot'),
        'right_avg_annot':osp.join(fsavg_subj_dir,'label','rh.aparc.annot')
    }


    # Find vertex indices on subject's pial surface
    pial_verts = [read_geometry(files['%s_pial'%h])[0] for h in hemispheres]

    distances = [dist.cdist(v,coords) for v in pial_verts]

    hemisphere = np.min(distances[0],0) < np.min(distances[1],0)
    pial_indices = [np.argmin(d,0) for d in distances]

    # Take those vertices in sphere.reg
    sphere_verts = [read_geometry(files['%s_sphere'%h])[0] for h in hemispheres]

    electrode_sphere_verts = [sv[pi] for (sv,pi) in zip(sphere_verts,pial_indices)]

    # Find indices of nearest vertices in fsaverage.?h.sphere.reg
    avg_sphere_verts = [read_geometry(files['%s_avg_sphere'%h])[0]
                                       for h in hemispheres]

    avg_sphere_indices = [np.argmin(dist.cdist(asv,esv),axis=0) for (asv,esv) in zip(avg_sphere_verts,electrode_sphere_verts)]
    # Take those vertices on average pial surface
    avg_pial_verts  = [read_geometry(files['%s_avg_pial'%h])[0] for h in hemispheres]

    avg_pial_inds,_,avg_pial_labels  =list(zip(*[read_annot(files['%s_avg_annot'%h])
                                       for h in hemispheres]))
    avg_pial_labels = [np.array(x) for x in avg_pial_labels]

    new_pial_verts = np.where(hemisphere[:,None],*[apv[asi] for apv,asi in zip(avg_pial_verts,avg_sphere_indices)])
    new_pial_labels = np.where(hemisphere, *[np.array(apl)[(api[asi])] for apl, api, asi in zip(avg_pial_labels,avg_pial_inds, avg_sphere_indices)])
    print((new_pial_verts.shape))
    print((new_pial_labels.shape))
    return new_pial_verts,new_pial_labels


def insert_transformed_coordinates(localization, files):
    Torig,Norig,talxfm = read_and_tx(files['coords_t1'], files['fs_orig_t1'],files['tal_xfm'], localization)
    localization.get_pair_coordinates('fs',pairs=localization.get_pairs(localization.get_lead_names()))
    localization.get_pair_coordinates('tal',pairs=localization.get_pairs(localization.get_lead_names()))
    localization.get_pair_coordinates('t1_mri',pairs=localization.get_pairs(localization.get_lead_names()))
    return Torig,Norig,talxfm


def invert_transformed_coords(localization,Torig,Norig,talxfm):
    for contact in localization.get_contacts():
        fs_corrected = localization.get_contact_coordinate('fs',contact,coordinate_type='corrected')
        coords = np.matrix([float(x) for x in fs_corrected[0]]+[1]).T
        mri_coords = Norig * inv(Torig) * coords
        tal_coords = talxfm * mri_coords
        mri_x = mri_coords[0]
        mri_y = mri_coords[1]
        mri_z = mri_coords[2]
        localization.set_contact_coordinate('t1_mri',contact,[mri_x,mri_y,mri_z],coordinate_type='corrected')
        localization.set_contact_coordinate('tal',contact,[tal_coords[i] for i in range(3)],coordinate_type='corrected')
    localization.get_pair_coordinates('tal')


def file_locations_fs(subject):
    """
    Creates the default file locations dictionary
    :param subject: Subject name to look for files within
    :returns: Dictionary of {file_name: file_location}
    """
    files = dict(
        coord_t1=os.path.join(paths.rhino_root, 'data10', 'RAM', 'subjects', subject, 'imaging', subject, 'electrodenames_coordinates_native_and_T1.csv'),
        fs_orig_t1=os.path.join(paths.rhino_root, 'data', 'eeg', 'freesurfer', 'subjects', subject, 'mri', 'orig.mgz')
    )
    return files

if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    import sys
    leads = build_leads_fs(file_locations(sys.argv[1]))
    leads_as_dict = leads_to_dict(leads)
    clean_dump(leads_as_dict, open(sys.argv[2],'w'), indent=2, sort_keys=True)

