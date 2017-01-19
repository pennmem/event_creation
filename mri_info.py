"""
Simple wrapper for freesurfer's mri_info function that returns a numpy matrix

Run:
    python mri_info.py <in_file> <transformtype>
to generate transformation matrices given a freesurfer orig.mgz file

"""
import os
import subprocess
import numpy as np
from config import paths
import logging

log = logging.getLogger('submission')

def get_transform(file, transform_type):
    """
    Executes mri_info and returns a numpy representation of the output
    >>> get_transform('/data/eeg/freesurfer/subjects/R1001P/mri/orig.mgz', 'vox2ras-tkr')

    :param file: path to mri file to read
    :param transform_type: type of transform to get from mri_info (such as 'vox2ras')
    :return: numpy array of transform matrix
    """
    mri_info_loc = os.path.join(paths.freesurfer_bin, 'mri_info')
    log.debug("Executing mri_info at {}".format(mri_info_loc))
    output = subprocess.check_output([mri_info_loc, file, '--{}'.format(transform_type)])
    num_output = [[float(x) for x in line.split()] for line in output.split('\n') if len(line)>0]
    return np.matrix(num_output)

if __name__ == '__main__':
    from config import RHINO_ROOT
    import sys
    infile = sys.argv[1]
    transformtype = sys.argv[2]
    print get_transform(infile, transformtype)
    """
    print get_transform(os.path.join(RHINO_ROOT, 'data/eeg/freesurfer/subjects/R1001P/mri/orig.mgz'), 'vox2ras')
    """
