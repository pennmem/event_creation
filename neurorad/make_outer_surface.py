import subprocess
from .config import paths
import logging
import os.path as osp
import os
import nibabel.freesurfer as nbfs
import scipy.ndimage as scimage
from skimage import measure
import numpy as np

log = logging.getLogger('submission')


def mri_fill(pial_file,filled_file):
    subprocess.call([osp.join(paths.freesurfer_bin,'mris_fill'),'-c','-r','1',pial_file,filled_file])


def make_outer_surface_matlab(filled_file,output_surface_file):
    assert osp.isfile(filled_file)
    # freesurfer_matlab = osp.join(osp.dirname(paths.freesurfer_bin),'matlab')
    freesurfer_matlab = '/home1/leond/electrode_vis/freesurfer/matlab'
    curr_dir = os.getcwd()
    print(('matlab directory: %s'%freesurfer_matlab))
    os.chdir(freesurfer_matlab)
    matlab_call = 'try make_outer_surface(\'{0}\',15,\'{1}\');catch; end; quit;'.format(filled_file,output_surface_file)
    bash_call = ['matlab', '-nodisplay', '-nojvm', '-nodesktop', '-r', "{}".format(matlab_call)]
    print(('system call: \n %s'%bash_call))
    subprocess.call(bash_call)
    os.chdir(curr_dir)


def extract_main_component(outer_surface_file,main_component_file):
    subprocess.call([osp.join(paths.freesurfer_bin,'mris_extract_main_component'), outer_surface_file,main_component_file])


def smooth_surface(main_component_file,smoothed_file):
    subprocess.call([osp.join(paths.freesurfer_bin,'mris_smooth'),'-nw','-n','30', main_component_file,smoothed_file])


def make_smoothed_surface_matlab(pial_surface_file):
    filled_file = pial_surface_file+'.filled.mgz'
    outer_surface_file  = pial_surface_file+'-outer'
    main_component_file = outer_surface_file+'-main'
    smoothed_file = outer_surface_file+'-smoothed'
    mri_fill(pial_surface_file,filled_file)
    make_outer_surface_matlab(filled_file,outer_surface_file)
    extract_main_component(outer_surface_file,main_component_file)
    smooth_surface(main_component_file,smoothed_file)

    return smoothed_file

def make_smoothed_surface(pial_surface_file,output_dir=''):
    psurface_base = osp.basename(pial_surface_file)
    filled_file = osp.join(output_dir,psurface_base+'.filled.mgz')
    outer_surface_file  = osp.join(output_dir,psurface_base+'-outer')
    main_component_file = outer_surface_file+'-main'
    smoothed_file = outer_surface_file+'-smoothed'
    mri_fill(pial_surface_file,filled_file)
    make_outer_surface(filled_file,outer_surface_file,12)
    extract_main_component(outer_surface_file,main_component_file)
    smooth_surface(main_component_file,smoothed_file)

    return smoothed_file


def make_outer_surface(filled_file,output_surface_file,se_diameter = 15):
    if os.path.isfile(output_surface_file):
        log.info('Dural surface mesh %s already exists'%os.path.basename(output_surface_file))
        return
    # read MRI
    volume = nbfs.MGHImage.from_filename(filled_file)
    volume = volume.get_data()

    # change elements from {0,1} to {0,255}
    volume *= 255

    # Gaussian filter (sigma=1mm)
    gaussian_volume = scimage.gaussian_filter(volume,1,mode='constant') # Is this correct?

    # Binarize filtered image
    avg = gaussian_volume.mean()
    gaussian_volume[gaussian_volume>avg]=255
    gaussian_volume[gaussian_volume<avg]=0

    # Morphological closing:

    # Construct structuring element
    xx,yy = np.meshgrid(list(range(-1*se_diameter+1,se_diameter)),list(range(-1*se_diameter+1,se_diameter)))
    se = (xx**2+yy**2)<se_diameter**2

    # Take closing

    closed_volume = np.stack([scimage.grey_closing(gaussian_volume[...,i],structure=se,mode='constant')
                             for i in range(gaussian_volume.shape[-1])],axis=-1)


    # Binarize closed image
    thresh = closed_volume.max()/2
    closed_volume[closed_volume<=thresh]=0
    closed_volume[closed_volume>thresh]=255

    # vertices,faces = isosurface(*,100)
    vertices,faces,_,_ = measure.marching_cubes(closed_volume,100)

    # Reorient
    v2 = np.zeros(vertices.shape)
    v2[:,0] = 129-vertices[:,0]
    v2[:,1] = vertices[:,2]-129
    v2[:,2] = 129-vertices[:,1]

    vertices= v2

    # Write geometry file
    nbfs.write_geometry(output_surface_file,vertices,faces)




