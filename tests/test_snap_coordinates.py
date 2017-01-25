"""
Test functions for snapping coordinates
calling:
    python test_snap_coordinates.py <subject>
will plot the original and snapped coordinates for that subject
"""

from snap_coordinates import *
import mayavi.mlab

def plot_coordinates(coordinates, color=(0,1,0)):
    """
    Plots a set of coordinates in 3d space
    :param coordinates: Nx3 matrix of 3d coordinates to plot
    :param color: 3-tuple of floats between 1 and 0. RGB values
    """
    mayavi.mlab.points3d(coordinates[:,0], coordinates[:,1], coordinates[:,2], color=color, scale_factor=3) 

def plot_surface(points, triangles):
    """
    Plots a surface in 3d space. 
    Points and triangles are the output of load_surface 
    (also the default output of nibabel's read_surf)
    :param points: Nx3 matrix containing coordinates of the vertices of the surface
    :param triangles: Nx3 matrix containing the indices of the connected vertices
    """
    mayavi.mlab.triangular_mesh(points[:,0], points[:,1], points[:,2], triangles, opacity=.4, color=(.7, .7, .7))

def plot_before_and_after(subject):
    """
    Plots electrodes before and after snapping for a given subject
    """
    # Gets the default file locations
    files = file_locations(subject)
    
    # Get and plot the electrode locations
    coordinates = get_raw_coordinates(files['raw_indiv'])
    plot_coordinates(coordinates)

    # Get and plot the surfaces
    points_l, faces_l = load_surface(files['surface_l'])
    plot_surface(points_l, faces_l)
    points_r, faces_r = load_surface(files['surface_r'])
    plot_surface(points_r, faces_r)
    
    # Snap the coordinates to either surface and plot
    new_coordinates = snap_to_surface(coordinates, np.concatenate((points_l, points_r)))
    plot_coordinates(new_coordinates, (0,0,1))

    # Show the plot, suspending execution until it is closed
    mayavi.mlab.show()
    
if __name__ == '__main__':
    import sys
    plot_before_and_after(sys.argv[1])
