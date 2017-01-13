"""
converts electrode (or for that matter, any point) coordinates in MRI space
to coordinates in freesurfer mesh space.
Requires subject ID to find coords data (in electrodenames_coordinates_native_and_T1.csv) and
the two matrices Norig and Torig which can be obtained using mri_info


Run:
    python voxcoords_to_fs.py <subject> <out_file>
to create a voxel_coordinates_fs.json file
"""
import re
import numpy as np
import os
from collections import defaultdict
from config import RHINO_ROOT
from mri_info import *
from numpy.linalg import inv
import json
from  localization import Localization


def xdot(*args):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """
    return reduce(np.dot, args)


def read_and_tx(t1_file, fs_orig_t1, leads):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """

    # Get freesurfer matrices
    Torig = get_transform(fs_orig_t1, 'vox2ras-tkr')
    Norig = get_transform(fs_orig_t1, 'vox2ras')
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
        # Make it a column vector
        coords[:, np.newaxis]
        
        # Compute the transformation
        fullmat = Torig * inv(Norig) 
        fscoords = fullmat.dot( coords )
        print "Transformed ", contact_name  

        fsx = fscoords[0,0]
        fsy = fscoords[0,1]
        fsz = fscoords[0,2]
        
        # Enter into "leads" dictionary
        leads.set_contact_coordinate('freesurfer', contact_name, [fsx, fsy, fsz], 'raw')
        leads.set_contact_coordinate('t1_mri', contact_name, [x, y, z])
    return leads


def build_leads_fs(files, leads):
    """
    Builds the leads dictionary from VOX_coords_mother and jacksheet
    :param files: dictionary of files including 'vox_mom' and 'jacksheet'
    :returns: dictionary of form {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """
    leads = read_and_tx(files['coord_t1'], files['fs_orig_t1'], leads)
    return leads

def file_locations_fs(subject):
    """
    Creates the default file locations dictionary
    :param subject: Subject name to look for files within
    :returns: Dictionary of {file_name: file_location}
    """
    files = dict(
        coord_t1=os.path.join(RHINO_ROOT, 'data10', 'RAM', 'subjects', subject, 'imaging', subject, 'electrodenames_coordinates_native_and_T1.csv'),
        fs_orig_t1=os.path.join(RHINO_ROOT, 'data', 'eeg', 'freesurfer', 'subjects', subject, 'mri', 'orig.mgz')
    )
    return files

if __name__ == '__main__':
    import sys
    leads = build_leads_fs(file_locations(sys.argv[1]))
    leads_as_dict = leads_to_dict(leads)
    clean_dump(leads_as_dict, open(sys.argv[2],'w'), indent=2, sort_keys=True)

