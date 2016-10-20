from vox_mother_converter import *

def show_subgrids(lead):
    grid_groups = set(contact.grid_group for contact in lead.values())
    for i, grid_group in enumerate(grid_groups):
        subgrid = {num:contact for num, contact in lead.items() if contact.grid_group == grid_group}
        for first_contact in subgrid.values(): break
        grid_size = first_contact.grid_size
        grid = np.empty((grid_size[1], grid_size[0]), dtype='S10')
        for j in range(len(grid)):
            grid[j] = ''
        for contact in subgrid.values():
            if not contact.grid_loc:
                print 'Cannot fit contact {} in grid!'.format(contact.name)
                continue
            grid[contact.grid_loc[1], contact.grid_loc[0]] = contact.name
        
        print '____{} {}____'.format(i, grid_size)
        print '\n'.join(['\t'.join(row) for row in grid])

def show_leads(leads):
    for name, lead in leads.items():
        print "********{}*********".format(name)
        show_subgrids(lead)

if __name__ == '__main__':
    import sys
    leads = build_leads(file_locations(sys.argv[1]))
    show_leads(leads)
