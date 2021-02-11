import  os.path as osp
class Paths:
    rhino_root = '/'
    freesurfer_bin = '/usr/global/freesurfer/bin'
    matlab_bin = '/usr/global/matlabR2015a/bin/matlab'
    ants_root = osp.join(osp.expanduser('~sudas'),'bin','ants')

paths = Paths()