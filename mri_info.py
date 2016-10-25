"""
Simple wrapper for freesurfer's mri_info function that returns a numpy matrix
"""
import os
import subprocess
import numpy as np

def get_transform(file, transform_type):
    """
    Executes mri_info and returns a numpy representation of the output
    >>> get_transform('/data/eeg/freesurfer/subjects/R1001P/mri/orig.mgz', 'vox2ras-tkr')

    :param file: path to mri file to read
    :param transform_type: type of transform to get from mri_info (such as 'vox2ras')
    :return: numpy array of transform matrix
    """

    output = subprocess.check_output(['mri_info', file, '--{}'.format(transform_type)])
    num_output = [[float(x) for x in line.split()] for line in output.split('\n') if len(line)>0]
    return np.array(num_output)

if __name__ == '__main__':
    from config import RHINO_ROOT
    print get_transform(os.path.join(RHINO_ROOT, 'data/eeg/freesurfer/subjects/R1001P/mri/orig.mgz'), 'vox2ras')
