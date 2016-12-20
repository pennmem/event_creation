import json
import numpy as np

class Contacts(object):

    def __init__(self, voxel_coordinates_json=None):
        self._orig_filename = voxel_coordinates_json
        if voxel_coordinates_json is not None:
            self._contact_dict = json.load(open(voxel_coordinates_json))
        else:
            self._contact_dict = None

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
        json.dump(self._contact_dict, open(json_file, 'w'))

    def get_lead_names(self):
        """ Returns list of strings of lead names """
        return self._contact_dict['leads'].keys()

    def get_lead_type(self, leads):
        """
        Returns the type of contact ("S", "G", or "D") for a given contact or lead
        """
        if not isinstance(leads, list):
            return self._contact_dict['leads'][leads]['type']
        else:
            return [self._contact_dict['leads'][lead]['type'] for lead in leads]

    def get_contact_names(self, leads=None):
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

    def get_contact_coordinates(self, coordinate_space, contacts):
        """
        coord_type: one of "fs", "t1_mri", "ct_voxels", ...
        contacts: string or list of strings of contact names
        returns: np.array of coordinates for each contact
        """
        if not isinstance(contacts, list):
            contact = self._contact_dict_by_name(contacts)
            return np.array(contact['coordinate_spaces'][coordinate_space])
        coordinates = []
        for contact in contacts:
            contact_dict = self._contact_dict_by_name(contact)
            coordinates.append(contact_dict['coordinate_spaces'][coordinate_space])
        return np.array(coordinates)

    
    def add_contact_coordinate(self, coordinates, coordinate_space, contacts):
        """
        Sets the coordinates for a given contact and coordinate type
        """
        if not isinstance(contacts, list):
            contacts = [contacts]
            coordinates = [coordinates]
        for contact_name, coordinate in zip(contacts, coordinates):
            contact = self._contact_dict_by_name(contact_name)
            contact['coordinate_spaces'][coordinate_space] = coordinate


    def add_contact_label(self, atlas_name, contacts, labels):
        """
        Sets the label for a given contact
        """
        if not isinstance(contacts, list):
            contacts = [contacts]
            labels = [labels]

        for contact_name, label in zip(contacts, labels):
            contact = self._contact_dict_by_name(contact_name)
            contact['atlases'][atlas_name] = label

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

    def get_pair_coordinates(self, coordinate_space, pairs):
        """
        coord_type: one of "fs", "t1_mri", "ct_voxels", ...
        pairs: Nx2 list of contact names (output of get_pairs)
        returns: np.array of coordinates for each pair
        """
        if not isinstance(pairs[0], list):
            pairs = [pairs]
        coordinates = []
        for pair in pairs:
            contact_name1 = pair[0]
            contact_name2 = pair[1]
            coord1 = self.get_contact_coordinates(coordinate_space, contact_name1)
            coord2 = self.get_contact_coordinates(coordinate_space, contact_name2)
            coord = ( np.array(coord1) + np.array(coord2) ) / 2
            coordinates.append(coord)
        return np.array(coordinates)

    def add_pair_label(self, atlas_name, pairs, labels):
        """
        Sets an atlas label for a given pair or list of pairs
        """
        if not isinstance(pairs[0], list):
            pairs = [pairs]
            labels = [labels]
        for pair, label in zip(pairs, labels):
            pair_dict = self._pair_dict_by_name(pair)
            pair_dict['atlases'][atlas_name] = label


    def _contact_dict_by_name(self, contact_name):
        for lead in self._contact_dict['leads']:
            for contact in lead['contacts']:
                if contact['name'] == contact_name:
                    return contact
        return None

    def _pair_dict_by_name(self, pair_names):
        for lead in self._contact_dict['leads']:
            for pair in lead['pairs']:
                if pair[0] == pair_names[0] and pair[1] == pair_names[1]:
                    return pair
        return None

