"""
"Test" is a loose term...
This outputs the grids and strips as they are interpreted from VOX_coords mother.
Meant to be used to visualize which bipolar pairs will be made.

Run with:
    python test_vox_mother_converter <subject>
"""
from voxcoords_to_fs import *

def show_subgrids(lead):
    """
    Shows all of the sub-grids for a given lead in tab-delimited text format
    :param lead: dictionary of {contact_name: contact object} for a single lead
    """
    grid_groups = set(contact.grid_group for contact in list(lead.values()))
    for i, grid_group in enumerate(grid_groups):
        subgrid = {num:contact for num, contact in list(lead.items()) if contact.grid_group == grid_group}
        for first_contact in list(subgrid.values()): break
        grid_size = first_contact.grid_size
        grid = np.empty((grid_size[1], grid_size[0]), dtype='S10')
        for j in range(len(grid)):
            grid[j] = ''
        for contact in list(subgrid.values()):
            if not contact.grid_loc:
                print('Cannot fit contact {} in grid!'.format(contact.name))
                continue
            grid[contact.grid_loc[1], contact.grid_loc[0]] = contact.name
        
        print('____{} {}____'.format(i, grid_size))
        print('\n'.join(['\t'.join(row) for row in grid]))

def show_leads(leads):
    """
    Prints all leads in tab-delimited text format
    :param lead: dictionary of {lead_name : {contact_name1: contact1, contact_name2: contact2...}}
    """
    for name, lead in list(leads.items()):
        print("********{}*********".format(name))
        # show_subgrids(lead)

if __name__ == '__main__':
    import sys
    leads = build_leads(file_locations(sys.argv[1]))
    show_leads(leads)
