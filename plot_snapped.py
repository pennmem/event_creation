from snap_coordinates import load_surface, file_locations
from test_snap_coordinates import plot_coordinates
import mayavi.mlab

subject = sys.argv[1]
loc_file = sys.argv[2]

files = file_locations(subject)

contacts = loc.get_contacts()

coords = loc.get_contact_coordinates('fs', contacts)
coords_coorected = loc.get_contact_coordinates('fs', contacts, 'corrected')

plot_coordinates(coords)
plot_coordaintes(corrected)

mayavi.malab.show()
