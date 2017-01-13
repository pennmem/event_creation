"""
Reads in localization data from the autoloc processing and adds to the electrode database
to coordinates in freesurfer mesh space.
Requires subject ID to find localization data


Run:
    python add_locations.py <subject> <out_file>
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


def read_loc(native_loc, leads):
    """
    Reads electrodenames_coordinates_native_and_T1.csv, returning a dictionary of leads
    :param t1_file: path to electrodenames_coordinates_native_and_T1.csv file
    :returns: dictionary of form TODO {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """

    for line in open(native_loc):
        split_line = line.strip().split(',')

        # Contact name
        contact_name = split_line[0]

        # Contact localization
        contact_autoloc = split_line[1]

        # Split into whole brain/MTL atlas labels
        loc_list = contact_autoloc.strip().split('/')

        # Enter into "leads" dictionary
        leads.set_contact_label('whole_brain', contact_name, loc_list[0])
        print "Read whole brain localization for ", contact_name
        if len(loc_list) > 1:
          leads.set_contact_label('mtl', contact_name, loc_list[1])
          print "Read MTL localization for ", contact_name
    return leads


def add_autoloc(files, leads):
    """
    Builds the leads dictionary from VOX_coords_mother and jacksheet
    :param files: dictionary of files including 'vox_mom' and 'jacksheet'
    :returns: dictionary of form {lead_name: {contact_name1: contact1, contact_name2:contact2, ...}}
    """
    leads = read_loc(files['native_loc'], leads)
    return leads

def file_locations_loc(subject):
    """
    Creates the default file locations dictionary
    :param subject: Subject name to look for files within
    :returns: Dictionary of {file_name: file_location}
    """
    files = dict(
        native_loc=os.path.join(RHINO_ROOT, 'data10', 'RAM', 'subjects', subject, 'imaging', subject, 'electrodenames_coordinates_native.csv'),
    )
    return files

if __name__ == '__main__':
    import sys
    leads = build_leads_sd(file_locations(sys.argv[1]))
    leads_as_dict = leads_to_dict(leads)
    clean_dump(leads_as_dict, open(sys.argv[2],'w'), indent=2, sort_keys=True)

