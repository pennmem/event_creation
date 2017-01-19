"""
converts electrode (or for that matter, any point) coordinates in MRI space
to coordinates in freesurfer mesh space.
Requires subject ID to find coords data (in electrodenames_coordinates_native_and_T1.csv) and
the two matrices Norig and Torig which can be obtained using mri_info


Run:
    python voxcoords_to_fs.py <subject> <out_file>
to create a voxel_coordinates_fs.json file
"""

import logging
from mri_info import *
from numpy.linalg import inv
from config import paths

log = logging.getLogger('submission')

def xdot(*args):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """
    return reduce(np.dot, args)


def read_and_tx(t1_file, fs_orig_t1, localization):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """

    # Get freesurfer matrices
    Torig = get_transform(fs_orig_t1, 'vox2ras-tkr')
    Norig = get_transform(fs_orig_t1, 'vox2ras')
    log.debug("Got transform")
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
        coords = np.array([float(x), float(y), float(z), 1])
        
        # Compute the transformation
        fullmat = Torig * inv(Norig) 
        fscoords = fullmat.dot( coords )
        log.debug("Transforming {}".format(contact_name))

        fsx = fscoords[0,0]
        fsy = fscoords[0,1]
        fsz = fscoords[0,2]

        # Enter into "leads" dictionary
        localization.set_contact_coordinate('fs', contact_name, [fsx, fsy, fsz], 'raw')
        localization.set_contact_coordinate('t1_mri', contact_name, [x, y, z])

    log.debug("Done with transform")

def insert_transformed_coordinates(localization, files):
    read_and_tx(files['coords_t1'], files['fs_orig_t1'], localization)

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
    log.setLevel(logging.DEBUG)
    import sys
    leads = build_leads_fs(file_locations(sys.argv[1]))
    leads_as_dict = leads_to_dict(leads)
    clean_dump(leads_as_dict, open(sys.argv[2],'w'), indent=2, sort_keys=True)

