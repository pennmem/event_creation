import json
import numpy as np
from json_cleaner import clean_json_dump

class Localization(object):

    def __init__(self, json_file=None):
        self._orig_filename = json_file
        self._contact_dict = None
        if json_file is not None:
            self.from_json(json_file)

    def from_json(self, json_file):
        self._contact_dict = json.load(open(json_file))
        self._orig_filename = json_file
        for lead_name, lead in self._contact_dict['leads'].items():
            for contact in lead['contacts']:
                if 'atlases' not in contact:
                    contact['atlases'] = {}
            pair_names = self._calculate_pairs(lead_name)
            if 'pairs' not in lead:
                lead['pairs'] = []
            for pair_name in pair_names:
                if pair_name not in [pair['names'] for pair in lead['pairs']]:
                    pair = {'names': pair_name, 'atlases': {}}
                    lead['pairs'].append(pair)

    def to_json(self, json_file):
        clean_json_dump(self._contact_dict, open(json_file, 'w'), indent=2, sort_keys=True)

    def get_lead_names(self):
        """ Returns list of strings of lead names """
        return list(self._contact_dict['leads'].keys())

    def get_lead_type(self, leads):
        """
        Returns the type of contact ("S", "G", or "D") for a given contact or lead
        """
        return self._contact_dict['leads'][leads]['type']

    def get_lead_types(self, leads):
        return [self._contact_dict['leads'][lead]['type'] for lead in leads]

    def get_contacts(self, leads=None):
        """ 
        Returns list of strings of contacts
        for each lead if provided, otherwise all contacts
        """
        if leads is None:
            leads = self.get_lead_names()
        if not isinstance(leads, list):
            leads = [leads]
        contacts = []
        for lead in leads:
            lead_contacts = [contact['name'] for contact in self._contact_dict['leads'][lead]['contacts']]
            contacts.extend(lead_contacts)
        return contacts

    def get_contact_coordinate(self, coordinate_space, contact):
        """
        coord_type: one of "fs", "t1_mri", "ct_voxels", ...
        contacts: string or list of strings of contact names
        returns: np.array of coordinates for each contact
        """

        contact_dict = self._contact_dict_by_name(contact)
        if coordinate_space not in contact_dict['coordinate_spaces']:
            return None
        return np.array(contact_dict['coordinate_spaces'][coordinate_space])

    def get_contact_coordinates(self, coordinate_space, contacts):
        coordinates = []
        for contact in contacts:
            coordinates.append(self.get_contact_coordinate(coordinate_space, contact))
        return np.array(coordinates)
    
    def set_contact_coordinate(self, coordinate, coordinate_space, contact):
        """
        Sets the coordinates for a given contact and coordinate type
        """
        contact_dict = self._contact_dict_by_name(contact)
        contact_dict['coordinate_spaces'][coordinate_space] = [float(c) for c in coordinate]

    def set_contact_coordinates(self, coordinates, coordinate_space, contacts):

        for contact, coordinate in zip(contacts, coordinates):
            self.set_contact_coordinate(coordinate, coordinate_space, contact)

    def get_contact_label(self, atlas_name, contact):

        contact = self._contact_dict_by_name(contact)
        if atlas_name in contact['atlases']:
            return contact['atlases'][atlas_name]
        else:
            return None

    def get_contact_labels(self, atlas_name, contacts):
        labels = []
        for contact_name in contacts:
            labels.append(self.get_contact_label(atlas_name, contact_name))
        return labels

    def set_contact_label(self, atlas_name, contact, label):
        """
        Sets the label for a given contact
        """
        contact_dict = self._contact_dict_by_name(contact)
        contact_dict['atlases'][atlas_name] = label

    def set_contact_labels(self, atlas_name, contacts, labels):
        for contact_name, label in zip(contacts, labels):
            self.set_contact_label(atlas_name, contact_name, label)

    def get_pairs(self, leads=None):
        """
        Returns a list of 2-tuples of strings with
        contacts names that form a pair
        """
        if leads is None:
            leads = self.get_lead_names()
        if not isinstance(leads, list):
            leads = [leads]
        pair_names = []
        for lead_name in leads:
            lead = self._contact_dict['leads'][lead_name]
            pairs = lead['pairs']
            pair_names.extend([pair['names'] for pair in pairs])
        return pair_names

    def get_pair_coordinate(self, coordinate_space, pair):
        """
        coord_type: one of "fs", "t1_mri", "ct_voxels", ...
        pairs: Nx2 list of contact names (output of get_pairs)
        returns: np.array of coordinates for each pair
        """
        contact_name1 = pair[0]
        contact_name2 = pair[1]
        coord1 = self.get_contact_coordinate(coordinate_space, contact_name1)
        coord2 = self.get_contact_coordinate(coordinate_space, contact_name2)
        if coord1 is None or coord2 is None:
            return None
        coord = ( np.array(coord1) + np.array(coord2) ) / 2
        return np.array(coord)

    def get_pair_coordinates(self, coordinate_space, pairs):
        coords = []
        for pair in pairs:
            coords.append(self.get_pair_coordinate(coordinate_space, pair))
        return np.array(coords)


    def set_pair_label(self, atlas_name, pair, label):
        """
        Sets an atlas label for a given pair or list of pairs
        """
        pair_dict = self._pair_dict_by_name(pair)
        pair_dict['atlases'][atlas_name] = label

    def get_pair_label(self, atlas_name, pair):
        pair = self._pair_dict_by_name(pair)
        if atlas_name in pair['atlases']:
            return pair['atlases'][atlas_name]
        return None

    def get_pair_labels(self, atlas_name, pairs):
        return [self.get_pair_label(atlas_name, pair) for pair in pairs]

    def set_pair_labels(self, atlas_name, pairs, labels):
        """
        Sets an atlas label for a given pair or list of pairs
        """
        for pair, label in zip(pairs, labels):
            self.set_pair_label(atlas_name, pair, label)

    def _calculate_pairs(self, lead_name):
        pairs = []
        lead = self._contact_dict['leads'][lead_name]
        for group_i in range(lead['n_groups']):
            group_contacts = [contact for contact in lead['contacts'] if contact['grid_group'] == group_i]
            for contact1 in group_contacts:
                gl1 = contact1['grid_loc']
                contact_pairs = [(contact1['name'], contact2['name']) for contact2 in group_contacts if
                                 gl1[0] == contact2['grid_loc'][0] and gl1[1] + 1 == contact2['grid_loc'][1] or
                                 gl1[1] == contact2['grid_loc'][1] and gl1[0] + 1 == contact2['grid_loc'][0]]
                pairs.extend(contact_pairs)
        return pairs

    def _contact_dict_by_name(self, contact_name):
        for lead in self._contact_dict['leads'].values():
            for contact in lead['contacts']:
                if contact['name'] == contact_name:
                    return contact
        raise Exception("Contact {} does not exist!".format(contact_name))

    def _pair_dict_by_name(self, pair_names):
        for lead in self._contact_dict['leads'].values():
            for pair in lead['pairs']:
                if pair['names'][0] == pair_names[0] and pair['names'][1] == pair_names[1]:
                    return pair
        return Exception("Pair {} does not exist!".format(pair_names))

if __name__ == '__main__':
    from pprint import pprint
    loc = Localization('sample_voxel_coordinates.json')
    pprint(loc._contact_dict)
    loc.to_json('sample_voxel_coordinates_out.json')

    leads = loc.get_lead_names()
    print('Lead names are {}'.format(leads))

    type1 = loc.get_lead_type(leads[0])
    print('Lead {} has type {}'.format(leads[0], type1))
    types_multi = loc.get_lead_types(leads[0:3])
    print('Leads {} have types {}'.format(leads[0:3], types_multi))

    contact_names1 = loc.get_contacts(leads[0])
    print('Lead {} has contacts {}'.format(leads[0], contact_names1))
    contact_names_multi = loc.get_contacts(leads[0:3])
    print('Leads {} have contacts {}'.format(leads[0:3], contact_names_multi))
    contact_names_all = loc.get_contacts()
    print('All contacts are {}'.format(contact_names_all))

    coords1 = loc.get_contact_coordinate('ct_voxel', contact_names1[0])
    print('Lead {} has coordinate {}'.format(contact_names1[0], coords1))
    coords_multi = loc.get_contact_coordinates('ct_voxel', contact_names_multi[0:3])
    print('Leads {} have coordinates {}'.format(contact_names_multi[0:3], coords_multi))
    coords_all = loc.get_contact_coordinates('ct_voxel', contact_names_all)
    print('All contacts have coordinates {}'.format(coords_all))

    fake_coords = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print('Setting contact {} coordinates to {}'.format(contact_names1[0], fake_coords[0]))
    loc.set_contact_coordinate(fake_coords[0], 'ct_voxel', contact_names1[0])
    coords1 = loc.get_contact_coordinate('ct_voxel', contact_names1[0])
    print('Contact {} now has coordinates {}'.format(contact_names1[0], coords1))
    if (fake_coords[0] != coords1).any():
        raise Exception("Coordinates do not match")

    print('Setting contacts {} coordinates to {}'.format(contact_names1[1:4], fake_coords[1:4]))

    loc.set_contact_coordinates(fake_coords[1:4], 'ct_voxel', contact_names1[1:4])
    coords_multi = loc.get_contact_coordinates('ct_voxel', contact_names1[1:4])
    print('Contacts {} now have coordinates {}'.format(contact_names1[1:4], coords_multi))
    if (fake_coords[1:4] != coords_multi).any():
        raise Exception("Coordinates do not match")

    empty_label = loc.get_contact_label('my_atlas', contact_names1[0])
    if empty_label is not None:
        raise Exception("Label not None before it is set")
    print("Contact {} has no label for atlas {}".format(contact_names1[0], 'my_atlas'))
    print("Setting contact {} atlas {} label to {}".format(contact_names1[0], 'my_atlas', 'in_the_brain'))
    loc.set_contact_label('my_atlas', contact_names1[0], 'in_the_brain')
    label = loc.get_contact_label('my_atlas', contact_names1[0])
    print("Contact {} atlas {} label is now {}".format(contact_names1[0], 'my_atlas', label))
    if (label != 'in_the_brain'):
        raise Exception("Label not properly set")
    labels_multi = loc.get_contact_labels('my_atlas', contact_names1[0:3])
    print("Contacts {} have labels {} for atlas {}".format(contact_names1[0:3], labels_multi, 'my_atlas'))
    new_labels = ['on_the_brain', 'off_the_brain']
    print("setting contacts {} atlas {} labels to {}".format(contact_names1[1:3], 'my_atlas', new_labels))
    loc.set_contact_labels('my_atlas', contact_names1[1:3], new_labels)
    labels_multi = loc.get_contact_labels('my_atlas', contact_names1[0:3])
    print("Contacts {} now have atlas {} labels {}".format(contact_names1[0:3], 'my_atlas', labels_multi))

    pairs_1 = loc.get_pairs(leads[0])
    print("Lead {} has pairs {}".format(leads[0], pairs_1))
    pairs_all = loc.get_pairs()
    print("All pairs in localization are {}".format(pairs_all))

    coords1 = loc.get_pair_coordinate('ct_voxel', pairs_1[0])
    print("Pair {} has coordinates {}".format(pairs_1[0], coords1))
    coords_multi = loc.get_pair_coordinates('ct_voxel', pairs_1[0:3])
    print("Pairs {} have coordinates {}".format(pairs_1[0:3], coords_multi))

    empty_label = loc.get_pair_label('my_atlas', pairs_all[0])
    if empty_label is not None:
        raise Exception("Label is not None before it is set")
    print("Pair {} has no label for atlas {}".format(pairs_all[0], "my_atlas"))
    print("Setting pairs {} atlas {} label to {}".format(pairs_all[0], 'my_atlas', 'by_the_brain'))
    loc.set_pair_label('my_atlas', pairs_all[0], 'by_the_brain')
    label = loc.get_pair_label('my_atlas', pairs_all[0])
    print("Pair {} atlas {} label now set to {}".format(pairs_all[0], 'my_atlas', label))
    new_labels = ['near_the_brain', 'under_the_brain']
    print("Setting pairs {} atlas {} labels to {}".format(pairs_all[1:3], 'my_atlas', new_labels))
    loc.set_pair_labels('my_atlas', pairs_all[1:3], new_labels)
    labels = loc.get_pair_labels('my_atlas', pairs_all[0:4])
    print("Pairs {} now have atlas {} labels {}".format(pairs_all[0:4], 'my_atlas', labels))

    loc.to_json("sample_voxel_coordinates_modified.json")
