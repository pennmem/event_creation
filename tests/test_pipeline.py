"""
"Test" is a loose term...
This outputs the grids and strips as they are interpreted from VOX_coords mother.
Meant to be used to visualize which bipolar pairs will be made.

Run with:
    python test_vox_mother_converter <subject>
"""

"""
from voxcoords_to_fs import *
"""
from vox_mother_converter import *
from calculate_transformation import *
from add_locations import *
from  localization import Localization

def show_subgrids(lead):
    """
    Shows all of the sub-grids for a given lead in tab-delimited text format
    :param lead: dictionary of {contact_name: contact object} for a single lead
    """
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
    """
    Prints all leads in tab-delimited text format
    :param lead: dictionary of {lead_name : {contact_name1: contact1, contact_name2: contact2...}}
    """
    for name, lead in leads.items():
        print "********{}*********".format(name)
        # show_subgrids(lead)

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', dest='subject', required=True)
    parser.add_argument('-o', '--output', dest='output', required=True)
    parser.add_argument('--add-fs', dest='add_fs', required=False, action='store_true', default=False)
    args = parser.parse_args()
    files_mother = file_locations(args.subject)
    if os.path.isfile(files_mother['vox_mom']) == False:
      print '\nVOX coords not available\n'
      exit(1)
    leads = build_leads(files_mother, args.add_fs)
    show_leads(leads)
    leads_as_dict = leads_to_dict(leads)
    clean_json_dump(leads_as_dict, open(args.output,'w'), indent=2, sort_keys=True)

    print '\nStage 0: json file saved with vox coords\n'

    """
    Add test for fs coordinate
    """

    leads_fs = Localization(args.output)
    files_fs = file_locations_fs(args.subject)
    if not os.path.isfile(files_fs["coord_t1"]) or not os.path.isfile(files_fs["fs_orig_t1"]):
      print '\n\nCoregistration not available\n'
      exit(1)
    leads_fs = build_leads_fs(files_fs, leads_fs)

    leads_fs.to_json(args.output + '_fs')
    print '\n\nStage 1: json file saved with freesurfer space coordinates\n'

    """
    Add test for localization info
    """
    leads_loc = Localization(args.output + '_fs')
    files_loc = file_locations_loc(args.subject)
    if not os.path.isfile(files_loc["native_loc"]):
      print '\n\nLocalization not available\n'
      exit(1)
    leads_loc = add_autoloc(files_loc, leads_loc)

    leads_loc.to_json(args.output + '_fs' + '_loc')
    print '\n\nStage 3: json file saved with localization information\n'

    """
    Add test for MNI info
    """
    leads_mni = Localization(args.output + '_fs' + '_loc')
    files_mni = file_locations_loc(args.subject)
    if not os.path.isfile(files_loc["mni_loc"]):
      print '\n\nMNI coordinates not available\n'
      exit(1)
    leads_mni = add_mni(files_mni, leads_mni)

    leads_mni.to_json(args.output + '_fs' + '_loc' + '_mni')
    print '\n\nStage 4: json file saved with MNI coordinates\n'

