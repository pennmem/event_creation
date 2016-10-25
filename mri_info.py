import subprocess
import numpy as np

def mri_info(file, type):
    output = subprocess.check_output(['mri_info', file, '--{}'.format(type)])
    num_output = [[float(x) for x in line.split()] for line in output.split('\n') if len(line)>0]
    return np.array(num_output)

if __name__ == '__main__':
     print mri_info('/data/eeg/freesurfer/subjects/R1001P/mri/orig.mgz', 'vox2ras')
