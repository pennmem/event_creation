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
from .config import RHINO_ROOT
from .json_cleaner import clean_dump
from .mri_info import *
from numpy.linalg import inv
import json
from functools import reduce

class Contact(object):
    """
    Simple Contact class that's just a container to hold properties of the contact.
    """

    def __init__(self, contact_name=None, contact_num=None, coords=None, fscoords=None, ):
        self.name = contact_name
        self.coords = coords
        self.fscoords = fscoords
        self.num = contact_num
    
    def to_dict(self):
        return {'mricoordinates': self.coords,
                'fscoordinates': self.fscoords}

def xdot(*args):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """
    return reduce(np.dot, args)

def leads_to_dict(leads):
    """
    Converts a dictionary that contains Contact objects to a dictionary that can
    be output as JSON.
    Copies all contacts into a single-level dictionary
    :param leads: dictionary of form {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    :returns: dictionary of form {contact_name1: {contact properties...}, contact_name2: {contact properties}}
    """
    out_dict = {}
    for lead_name, contacts in list(leads.items()):
        for contact in list(contacts.values()):
            out_dict[contact.name] = contact.to_dict()
    return out_dict


def read_and_tx(t1_file, fs_orig_t1):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """

    # Get freesurfer matrices
    Torig = get_transform(fs_orig_t1, 'vox2ras-tkr')
    Norig = get_transform(fs_orig_t1, 'vox2ras')
    # defaultdict allows dictionaries to be created on access so we don't have to check for existence
    leads = defaultdict(dict)
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
        print(contact_name, fscoords)  

        # Split out contact name and number
        match = re.match(r'(.+?)(\d+$)', contact_name)
        lead_name = match.group(1)
        contact_num = int(match.group(2))

        # Enter into "leads" dictionary
        contact = Contact(contact_name, contact_num, coords, fscoords)
        leads[lead_name][contact_num] = contact
    return leads


def build_leads(files):
    """
    Builds the leads dictionary from VOX_coords_mother and jacksheet
    :param files: dictionary of files including 'vox_mom' and 'jacksheet'
    :returns: dictionary of form {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """
    leads = read_and_tx(files['coord_t1'], files['fs_orig_t1'])
    return leads

def file_locations(subject):
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
    leads = build_leads(file_locations(sys.argv[1]))
    leads_as_dict = leads_to_dict(leads)
    clean_dump(leads_as_dict, open(sys.argv[2],'w'), indent=2, sort_keys=True)

