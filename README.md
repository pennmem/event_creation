# neruorad_pipeline

To install dependencies:

Install anaconda or miniconda (https://www.continuum.io/downloads)
    conda install scipy
    conda install mayavi
    pip install nibabel
    
I think that's all, but I may be forgetting a depedency.

After installation (if not on rhino) set the path to mounted rhino in config.py

To view coordinate snapping:

    python test_snap_coordinates.py <subject> 


There are four main files in this repo (so far):

* snap\_coordinates.py
 * handles the snapping of coordinates to a surface (pial surface, by default)
* vox\_mother\_converter.py
 * Converts VOX\_coords\_mother.txt into voxel\_coordinates.json 
* mri\_info.py
 * Defines one function (mri\_info.py) that is a wrapper to call the freesurfer mri\_info function
* json\_cleaner.py
 * Used in the output of vox\_mother\_converter to output json on a single line. 

