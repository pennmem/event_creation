from snap_coordinates import load_surface, file_locations
from test_snap_coordinates import plot_coordinates, plot_surface
from localization import Localization
import mayavi.mlab
import sys

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
