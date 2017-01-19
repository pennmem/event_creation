from localization import Localization
import mayavi.mlab
import sys

def load_surface(path_to_surface):
    """
    Loads a surface file from disk with nibabel functionality,
    and returns a list of the points and faces in that file
    :param path_to_surface: Path to lh.pial (or other surface file) 
    :return: (vertices, faces)
    """
    points, faces = nib.freesurfer.io.read_geometry(path_to_surface)
    return points, faces


def file_locations(subject):
    """
    Constructs a dictionary of file locations. 
    This is a stand-in for the submission utility.
    Not really necessary at the moment
    """
    
    files = dict(
        raw_indiv=os.path.join(RHINO_ROOT, 'data', 'eeg', subject, 'tal', 'RAW_coords_indivSurf.txt'),
        surface_r=os.path.join(RHINO_ROOT, 'data', 'eeg', 'freesurfer', 'subjects', subject, 'surf', 'rh.pial'),
        surface_l=os.path.join(RHINO_ROOT, 'data', 'eeg', 'freesurfer', 'subjects', subject, 'surf', 'lh.pial')
    )
    return files


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


subject = sys.argv[1]
loc_file = sys.argv[2]

files = file_locations(subject)

loc = Localization(loc_file)
contacts = loc.get_contacts()

coords = loc.get_contact_coordinates('fs', contacts)
coords_corrected = loc.get_contact_coordinates('fs', contacts, 'corrected')

plot_coordinates(coords)
plot_coordinates(coords_corrected, (0, 0, 1))

p, f = load_surface(files['surface_l'])
p2, f2 = load_surface(files['surface_r'])
plot_surface(p2, f2)
plot_surface(p, f)

mayavi.mlab.show()
