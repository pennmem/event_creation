"""
Library to handle snapping coordinates to an mri surface
(defaults to pial surface, but customizable)
"""
import os
from config import RHINO_ROOT
import numpy as np
import nibabel as nib
from scipy import spatial

def get_raw_coordinates(filename):
    """
    Gets coordinates from a "RAW_coords" file
    Meant to be used with the RAW_coords_indivSurf file
    to get the coordinates corresponding to the individual surface
    :return: numpy array of coordinates
    """
    contents = [l.split() for l in open(filename).readlines()]
    coords = [(float(line[1]), float(line[2]), float(line[3])) for line in contents]
    return np.array(coords)

def load_surface(path_to_surface):
    points, faces = nib.freesurfer.io.read_geometry(path_to_surface)
    return points, faces

def snap_to_surface(coordinates, surface_points):
    tree = spatial.KDTree(surface_points)
    dists, indices = tree.query(coordinates)
    return surface_points[indices]

def load_and_snap(files):
    coordinates = get_raw_coordinates(files['raw_indiv'])

    points_l, faces_l = load_surface(files['surface_l'])
    points_r, faces_r = load_surface(files['surface_r'])
    new_coordinates = snap_to_surface(coordinates, np.concatenate((points_l, points_r)))

def file_locations(subject):
    files = dict(
        raw_indiv=os.path.join(RHINO_ROOT, 'data', 'eeg', subject, 'tal', 'RAW_coords_indivSurf.txt'),
        surface_r=os.path.join(RHINO_ROOT, 'data', 'eeg', 'freesurfer', 'subjects', subject, 'surf', 'rh.pial'),
        surface_l=os.path.join(RHINO_ROOT, 'data', 'eeg', 'freesurfer', 'subjects', subject, 'surf', 'lh.pial')
    )
    return files

if __name__ == '__main__':
    import sys
    load_and_snap(file_locations(sys.argv[1]))


