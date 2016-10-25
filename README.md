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
 * Test by making plot with: `test_snap_coordinates.py <subject>`

* vox\_mother\_converter.py
 * Converts VOX\_coords\_mother.txt into voxel\_coordinates.json 
 * Make voxel\_coordintes.json with: `python vox_mother_converter.py <subject> <output_file>`
 * View grids for debugging with: `python test_vox_mother_converter.py <subject>`

* mri\_info.py
 * Defines one function (mri\_info.py) that is a wrapper to call the freesurfer mri\_info function
 * Use with:
    ``` 
    >>> from mri_info import get_transform
    >>> get_transform('/path/to/surface.mgz', 'vox2ras')`
    ```
 * View sample output with: `python mri_info.py`
* json\_cleaner.py
 * Used in the output of vox\_mother\_converter to output json on a single line. 

