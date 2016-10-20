from snap_coordinates import *
import mayavi.mlab

def plot_coordinates(coordinates, color=(0,1,0)):
    x = mayavi.mlab.points3d(coordinates[:,0], coordinates[:,1], coordinates[:,2], color=color, scale_factor=3) 

def plot_surface(points, triangles):
    mayavi.mlab.triangular_mesh(points[:,0], points[:,1], points[:,2], triangles, opacity=.4, color=(.7, .7, .7))

def plot_before_and_after(subject):
    files = file_locations(subject)

    coordinates = get_raw_coordinates(files['raw_indiv'])
    plot_coordinates(coordinates)

    points_l, faces_l = load_surface(files['surface_l'])
    plot_surface(points_l, faces_l)
    points_r, faces_r = load_surface(files['surface_r'])
    plot_surface(points_r, faces_r)

    new_coordinates = snap_to_surface(coordinates, np.concatenate((points_l, points_r)))
    plot_coordinates(new_coordinates, (0,0,1))

    mayavi.mlab.show()
    

if __name__ == '__main__':
    import sys
    plot_before_and_after(sys.argv[1])
