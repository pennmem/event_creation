import json
import numpy as np
from .json_cleaner import clean_json_dump, clean_json_dumps
from itertools import combinations
from .version import __version__


class InvalidFieldException(Exception):
    pass


class InvalidContactException(Exception):
    pass


def merge_repeated_keys(pairs):
    d = {}
    for (k,v) in pairs:
        if k in d:
            if isinstance(d[k],dict) and isinstance(v,dict):
                d[k] = merge_repeated_keys(list(d[k].items())+list(v.items()))
            elif isinstance(d[k],list) and isinstance(v,list):
                d[k].extend(v)
            else:
                d[k]=v
        else:
            d[k]=v

    return d


class Localization(object):

    VALID_COORDINATE_SPACES = (
        'ct_voxel',
        'fs',
        't1_mri',
        't2_mri',
        'mni',
		'tal',
        'fsaverage',
        'hcp',
    )
    
    VALID_COORDINATE_TYPES = (
        'raw',
        'corrected'
    )

    VALID_INFO = (
        'displacement',
        'closest_vertex_distance',
        'closest_vertex_coordinate',
        'fsaverage_vertex_coordinate',
        'closest_ortho_vertex_coordinate',
        'linked_electrodes',
        'link_displaced',
        'group_corrected',
    )

    VALID_ATLASES = (
        'dk',
        'dkavg',
        'hcp',
        'whole_brain',
        'mtl',
        'manual',
    )

    def __init__(self, json_file=None):
        self._orig_filename = json_file
        self._contact_dict = {}
        if json_file is not None:
            self.from_json(json_file)
            self.get_pair_coordinates('ct_voxel',self.get_pairs(self.get_lead_names()))
        if not self._contact_dict.get('version'):
            self._contact_dict['version'] = __version__


    @property
    def version(self):
        return self._contact_dict['version']

    def from_json(self, json_file):
        """ Loads from a json file """
        self._contact_dict = json.load(open(json_file),object_pairs_hook=merge_repeated_keys)
        self._orig_filename = json_file
        for lead_name, lead in list(self._contact_dict['leads'].items()):
            for contact in lead['contacts']:
                for field in contact:
                    # Naming convention for voxTool compatibility
                    if isinstance(field,str) and 'grid' in field:
                        new_field = field.replace('grid','lead')
                        contact[new_field] = contact[field]
                        del contact[field]
                if 'atlases' not in contact:
                    contact['atlases'] = {}
                if 'info' not in contact:
                    contact['info'] = {}

            pair_names = self._calculate_pairs(lead_name)
            if 'pairs' not in lead:
                lead['pairs'] = []
            for pair_name in pair_names:
                if pair_name not in [type(pair_name)(pair['names']) for pair in lead['pairs']]:
                    pair = {'names': pair_name, 'atlases': {}, 'info': {},'coordinate_spaces':{}}
                    lead['pairs'].append(pair)
        return

    def to_json(self, json_file):
        """ Dumps to a json file """
        clean_json_dump(self._contact_dict, open(json_file, 'w'), indent=2, sort_keys=True)

    def to_jsons(self):
        return clean_json_dumps(self._contact_dict, indent=2, sort_keys=True)

    def to_vox_mom(self,fname):
        csv_out = []
        for lead in sorted(self.get_lead_names(),cmp=lambda x,y:cmp(x.upper(),y.upper()) ):
            ltype = self.get_lead_type(lead)
            n_groups = self._contact_dict['leads'][lead]['n_groups']
            n_contacts = len(self.get_contacts(lead))
            for contact in sorted(self.get_contacts(leads=lead)):
                voxel = self.get_contact_coordinate('ct_voxel',contact)
                csv_out += "%s\t%s\t%s\t%s\t%s\t%s %s\n"%(
                    contact.upper(),voxel[0,0],voxel[0,1],voxel[0,2],ltype,n_contacts,n_groups
                )
        with open(fname,'w') as vox_mom:
            vox_mom.writelines(csv_out)


    def get_lead_names(self):
        """ Returns list of strings of lead names """
        return list(self._contact_dict['leads'].keys())

    def get_lead_type(self, lead):
        """ Gets the type of the lead ("S", "G","D","uD","uW")
        :param lead: the name of a lead
        :return: "S", "G","D","uD", or "uW"
        """
        if lead in self._contact_dict['leads']:
            return self._contact_dict['leads'][lead]['type']
        else:
            raise InvalidContactException("Lead {} does not exist".format(lead))

    def get_lead_types(self, leads):
        """ Gets the types for each lead
        :param leads: a list of lead names
        :returns: a list of types ("S", "G", or "D")
        """
        return [self.get_lead_type(lead) for lead in leads]

    def get_contact_type(self, contact):
        """ Gets the type of a contact
        :param contact: The name of the contact
        :returns: "S", "G" or "D"
        """
        for lead in list(self._contact_dict['leads'].values()):
            for contact_dict in lead['contacts']:
                if contact_dict['name'] == contact:
                    return lead['type']
        raise InvalidContactException("Contact {} does not exist".format(contact))

    def get_contact_types(self, contacts):
        """ Gets the types for each provided contact
        :param contacts: A list of contact names
        :returns: a list of types ("S", "G", or "D")
        """
        return [self.get_contact_type(contact) for contact in contacts]

    def get_contacts(self, leads=None):
        """ Gets the names of contacts
        :param leads: (optional) the lead or leads on which to find the contacts. If not provided finds on all leads
        :return: a list of contact names
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

    def get_contact_coordinate(self, coordinate_space, contact, coordinate_type='raw'):
        """ Gets the coordinates for a single contact
        :param coordinate_space: one of "fs", "t1_mri", "ct_voxels", ...
        :param contact: name of the contact
        :returns: np.array of [x,y,z] for contact
        """
        self._validate_space(coordinate_space)
        self._validate_type(coordinate_type)
        contact_dict = self._contact_dict_by_name(contact)
        try:
            return np.array(contact_dict['coordinate_spaces'][coordinate_space][coordinate_type], ndmin=2)
        except KeyError:
            output = np.empty((1, 3))
            output[:] = np.NAN
            return output

    def get_contact_coordinates(self, coordinate_space, contacts=None, coordinate_type='raw'):
        """ Gets the coordinates for all provided contacts
        :param coordinate_space: one of "fs", "t1_mri", "ct_voxels"...
        :param contacts: list of contact names
        :returns: np.array of [[x1, y1, z1], [x2, y2, z2], ...] for each contact
        """
        if contacts is None:
            contacts = self.get_contacts()
        coordinates = np.array([[], [], []]).T
        for contact in contacts:
            coordinate = self.get_contact_coordinate(coordinate_space, contact, coordinate_type)
            coordinates = np.concatenate((coordinates, coordinate), 0)
        return np.array(coordinates)
    
    def set_contact_coordinate(self, coordinate_space, contact, coordinate, coordinate_type='raw'):
        """ Sets the coordinates for a given contact and coordinate space
        :param coordinate: [x, y, z] of this contact's coordinates
        :param coordinate_space: one of "fs", "t1_mri", "ct_voxels"...
        :param contact: name of contact to modify
        """
        self._validate_space(coordinate_space)
        self._validate_type(coordinate_type)
        contact_dict = self._contact_dict_by_name(contact)
        if coordinate_space not in contact_dict['coordinate_spaces']:
            contact_dict['coordinate_spaces'][coordinate_space] = {}
        contact_dict['coordinate_spaces'][coordinate_space][coordinate_type] = [float(c) for c in coordinate]

    def set_contact_coordinates(self, coordinate_space, contacts, coordinates, coordinate_type='raw'):
        """ Sets the coordinates for all provided contacts
        :param coordinates: [[x1, y1, z1], [x2, y2, z2], ...] for each contact
        :param coordinate_space: one of "fs", "t1_mri", "ct_voxels"...
        :param contacts: list of contact names (same length as coordinates)
        """
        for contact, coordinate in zip(contacts, coordinates):
            self.set_contact_coordinate(coordinate_space, contact, coordinate, coordinate_type)

    def get_contact_label(self, atlas_name, contact):
        """ Gets the atlas label for a given contact
        :param atlas_name: name of atlas to look up
        :param contact: name of contact to look up
        :return: atlas label if atlas exists for this contact, otherwise None
        """
        self._validate_atlas(atlas_name)
        contact = self._contact_dict_by_name(contact)
        if atlas_name in contact['atlases']:
            return contact['atlases'][atlas_name]
        else:
            return None

    def get_contact_labels(self, atlas_name, contacts):
        """ Gets the atlas labels for each provided contact
        :param atlas_name: name of atlas to look up
        :param contacts: list of strings of contact names
        :return: [label1, label2, ...] for each contact. Label is set to None if not yet assigned
        """
        labels = []
        for contact_name in contacts:
            labels.append(self.get_contact_label(atlas_name, contact_name))
        return labels

    def set_contact_label(self, atlas_name, contact, label):
        """Sets the atlas label for a given contact
        :param atlas_name: name of the atlas to set
        :param contact: Name of the contact to set
        :param label: atlas label to apply
        """
        self._validate_atlas(atlas_name)
        contact_dict = self._contact_dict_by_name(contact)
        contact_dict['atlases'][atlas_name] = label

    def set_contact_labels(self, atlas_name, contacts, labels):
        """ Sets the atlas labels for a list of contacts
        :param atlas_name: name of the atlas to set
        :param contacts: list of names of the contacts to change
        :param labels: list of labels, same size as contacts
        """
        for contact_name, label in zip(contacts, labels):
            self.set_contact_label(atlas_name, contact_name, label)
    
    def get_contact_info(self, info_label, contact):
        """ Gets the info field of a specific contact
        :param info_label: label of the information to get ('displacement', e.g.)
        :param contact: name of contact to get
        """
        self._validate_info(info_label)
        contact_dict = self._contact_dict_by_name(contact)
        if info_label not in contact_dict['info']:
            return None
        return contact_dict['info'][info_label]

    def get_contact_infos(self, info_label, contacts):
        """ Gets the info field of a set of contacts
        :param info_label: label of the information to get ('displacement', e.g.)
        :param contact: list of names of contacts to get
        """
        info = []
        for contact_name in contacts:
            info.append(self.get_contact_info(info_label, contact_name))
        return info
    
    def set_contact_info(self, info_label, contact, info_value):
        """ Sets the info field of a contact
        :param info_label: label of the information to set ('displacement', e.g.)
        :param contact: name of the contact to set
        :param info_value: value to set info to
        """
        self._validate_info(info_label)
        contact_dict = self._contact_dict_by_name(contact)
        contact_dict['info'][info_label] = info_value

    def set_contact_infos(self, info_label, contacts, info_values):
        """ Sets the info field of multiple contacts
        :param info_label: label of the information to set ('displacement', e.g.)
        :param contacts: list of names of the contact to set
        :param info_values: list of values to set info to (same length as contacts)
        """
        for contact_name, info_value in zip(contacts, info_values):
            self.set_contact_info(info_label, contact_name, info_value)

    def get_pairs(self, leads=None):
        """ Gets the names of adjacent contacts
        :params leads: (optional) The lead or leads on which to find the pairs. If not provided finds all pairs.
        :returns: list of 2-tuples of contact names representing adjacent electrodes
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

    def set_pairs_coordinates(self,space,pairs,coordinates,type):
        for (p,c) in zip(pairs,coordinates):
            self.set_pair_coordinate(space,p,c,type)

    def set_pair_coordinate(self,space,pair_name,coordinate,type):
        pair_dict = self._pair_dict_by_name(pair_name)
        if 'coordinate_spaces' not in pair_dict:
            pair_dict['coordinate_spaces'] = {}
        if space not in pair_dict['coordinate_spaces']:
            pair_dict['coordinate_spaces'][space] = {}
        pair_dict['coordinate_spaces'][space][type] = [float(c) for c in coordinate.flat]

    def _get_pair_coordinate(self,space,pair_name,type):
        pair_dict = self._pair_dict_by_name(pair_name)
        coord = pair_dict['coordinate_spaces'][space][type]
        if coord  is not None:
            return coord
        raise KeyError

    def get_pair_coordinate(self, coordinate_space, pair, coordinate_type='raw'):
        """ Gets the coordinates at which a pair is located
        :param coordinate_space: one of "fs", "t1_mri", "ct_voxels"...
        :param pair: 2 element list containing the names of the contacts in the pair
        :returns: np.array of [x,y,z] for the provided pair (average of the provided coordinates)
        """
        self._validate_space(coordinate_space)
        self._validate_type(coordinate_type)
        try: 
            coord = self._get_pair_coordinate(coordinate_space,pair,coordinate_type)
        except (KeyError, InvalidContactException) as error:
            contact_name1 = pair[0]
            contact_name2 = pair[1]
            coord1 = self.get_contact_coordinate(coordinate_space, contact_name1, coordinate_type)
            coord2 = self.get_contact_coordinate(coordinate_space, contact_name2, coordinate_type)
            if coord1 is None or coord2 is None:
                return None
            coord = ( np.array(coord1) + np.array(coord2) ) / 2
            if type(error) != InvalidContactException:
                self.set_pair_coordinate(coordinate_space,pair,coord,coordinate_type)
        return np.array(coord)

    def get_pair_coordinates(self, coordinate_space, pairs=None, coordinate_type='raw'):
        """ Gets the coordinates at which a set of pairs are located
        :param coordinate_space: one of "fs", "t1_mri", "ct_voxels"...
        :param pairs: 2xN list containing the names of contacts in each pair
        :returns: np.array of [[x,y,z]] for the provided pairs
        """
        coords = []
        if pairs is None:
            pairs = self.get_pairs()
        coords = [self.get_pair_coordinate(coordinate_space,pair,coordinate_type) for pair in pairs]
        return np.array([c for c in coords if c is not None])


    def set_pair_label(self, atlas_name, pair, label):
        """ Sets an atlas label for a given pair
        :param atlas_name: one of "fs", "t1_mri", "ct_voxels"...
        :param pair: 2 element list with the names of the contacts in the pair
        :param label: atlas label to set
        """
        self._validate_atlas(atlas_name)
        pair_dict = self._pair_dict_by_name(pair)
        pair_dict['atlases'][atlas_name] = label

    def get_pair_label(self, atlas_name, pair):
        """ Gets an atlas label for a given pair
        :param atlas_name: one of "fs", "t1_mri", "ct_voxels"...
        :param pair: 2-element list with the names of the contacts in the pair
        :returns: atlas label for the pair, or None if atlas does not exist
        """
        self._validate_atlas(atlas_name)
        pair = self._pair_dict_by_name(pair)
        if atlas_name in pair['atlases']:
            return pair['atlases'][atlas_name]
        return None

    def get_pair_labels(self, atlas_name, pairs):
        """ Gets the atlas labels for a set of pairs
        :param atlas_name: one of "fs", "t1_mri", "ct_voxels"...
        :param pairs: 2xN list with the names of the contacts in the pairs
        :returns: list of atlas labels. Labels are None where pair does not have that atlas assigned
        """
        return [self.get_pair_label(atlas_name, pair) for pair in pairs]

    def set_pair_labels(self, atlas_name, pairs, labels):
        """ Sets the atlas labels for a set of pairs
        :param atlas_name: one of "fs", "t1_mri", "ct_voxels"...
        :param pairs: 2xN list with the names of the contacts in the pairs
        :param labels: N-element list with the proper atlas labels to set
        """
        for pair, label in zip(pairs, labels):
            self.set_pair_label(atlas_name, pair, label)
    
    def get_pair_info(self, info_label, pair):
        """ Gets the info field of a specific pair
        :param info_label: label of the information to get ('displacement', e.g.)
        :param pair: name of pair to get (2 element list)
        """
        self._validate_info(info_label)
        pair_dict = self._pair_dict_by_name(pair)
        if info_label not in pair_dict['info']:
            return None
        return pair_dict['info'][info_label]

    def get_pair_infos(self, info_label, pairs):
        """ Gets the info field of a set of pairs
        :param info_label: label of the information to get ('displacement', e.g.)
        :param pairs: list of names of pairs to get (2 x N list)
        """
        info = []
        for pair_name in pairs:
            info.append(self.get_pair_info(info_label, pair_name))
        return info
    
    def set_pair_info(self, info_label, pair, info_value):
        """ Sets the info field of a pair
        :param info_label: label of the information to set ('displacement', e.g.)
        :param pair: name of the pair to set (2 element list)
        :param info_value: value to set info to
        """
        self._validate_info(info_label)
        pair_dict = self._pair_dict_by_name(pair)
        pair_dict['info'][info_label] = info_value

    def set_pair_infos(self, info_label, pairs, info_values):
        """ Sets the info field of multiple pairs
        :param info_label: label of the information to set ('displacement', e.g.)
        :param pairs: list of names of the pairs to set (2 x N list)
        :param info_values: list of values to set info to (same length as pairs)
        """
        for pair_name, info_value in zip(pairs, info_values):
            self.set_pair_info(info_label, pair_name, info_value)

    # def add_pair(self,lead_name,pair_names):
    #     self._contact_dict['leads'][lead_name][]
    
    def _calculate_pairs(self, lead_name):
        pairs = []
        lead = self._contact_dict['leads'][lead_name]
        for group_i in range(lead['n_groups']):
            group_contacts = [contact for contact in lead['contacts'] if contact['lead_group'] == group_i]
            contact_pairs = [(contact1['name'],contact2['name']) for contact1,contact2 in combinations(group_contacts,2)
                             if is_adjacent(contact1['lead_loc'],contact2['lead_loc'])]
            pairs.extend(contact_pairs)
        return [sorted(x) for x in pairs]
    
    @classmethod
    def _validate_space(cls, coordinate_space):
        if not coordinate_space in cls.VALID_COORDINATE_SPACES:
            raise InvalidFieldException("Coordinate space {} is not valid. "
                                        "Options are: {}".format(coordinate_space,
                                                                 cls.VALID_COORDINATE_SPACES))

    @classmethod
    def _validate_type(cls, coordinate_type):
        if not coordinate_type in cls.VALID_COORDINATE_TYPES:
            raise InvalidFieldException("Coordinate type {} is not valid. "
                                        "Options are: {}".format(coordinate_type,
                                                                 cls.VALID_COORDINATE_TYPES))

    @classmethod
    def _validate_info(cls, info_label):
        if not info_label in cls.VALID_INFO:
            raise InvalidFieldException("Info type {} is not valid. "
                                        "Options are {}".format(info_label,
                                                                cls.VALID_INFO))

    @classmethod
    def _validate_atlas(cls, atlas_label):
        if not atlas_label in cls.VALID_ATLASES:
            raise InvalidFieldException("Atlas type {} is not valid. "
                                        "Options are {}".format(atlas_label,
                                                                cls.VALID_ATLASES))

    def _contact_dict_by_name(self, contact_name):
        for lead in list(self._contact_dict['leads'].values()):
            for contact in lead['contacts']:
                if contact['name'] == contact_name:
                    return contact
        raise InvalidContactException("Contact {} does not exist!".format(contact_name))

    def _pair_dict_by_name(self, pair_names):
        for lead in list(self._contact_dict['leads'].values()):
            for pair in lead['pairs']:
                if all(p in pair['names'] for p in pair_names):
                    return pair
        raise InvalidContactException("Pair {} does not exist!".format(pair_names))

def is_adjacent(loc1,loc2):
    diff= np.abs(np.subtract(loc1,loc2))
    return all(diff==np.array([1,0])) or all(diff==np.array([0,1]))


if __name__ == '__main__':
    from pprint import pprint
    import sys
    if len(sys.argv) > 1:
        loc = Localization(sys.argv[1])
    else:
        loc = Localization('sample_voxel_coordinates.json')
    pprint(loc._contact_dict)
    loc.to_json('sample_voxel_coordinates_start.json')

    leads = loc.get_lead_names()
    print(('Lead names are {}'.format(leads)))

    type1 = loc.get_lead_type(leads[0])
    print(('Lead {} has type {}'.format(leads[0], type1)))
    types_multi = loc.get_lead_types(leads[0:3])
    print(('Leads {} have types {}'.format(leads[0:3], types_multi)))

    contact_names1 = loc.get_contacts(leads[0])
    print(('Lead {} has contacts {}'.format(leads[0], contact_names1)))
    contact_names_multi = loc.get_contacts(leads[0:3])
    print(('Leads {} have contacts {}'.format(leads[0:3], contact_names_multi)))
    contact_names_all = loc.get_contacts()
    print(('All contacts are {}'.format(contact_names_all)))

    type1 = loc.get_contact_type(contact_names1[0])
    print(("Contact {} has type {}".format(contact_names1[0], type1)))
    types_multi = loc.get_contact_types(contact_names_all[:5])
    print(("Contacts {} have types {}".format(contact_names_all[:5], types_multi)))

    coords1 = loc.get_contact_coordinate('ct_voxel', contact_names1[0])
    print(('Lead {} has coordinate {}'.format(contact_names1[0], coords1)))
    coords_multi = loc.get_contact_coordinates('ct_voxel', contact_names_multi[0:3])
    print(('Leads {} have coordinates {}'.format(contact_names_multi[0:3], coords_multi)))
    coords_all = loc.get_contact_coordinates('ct_voxel', contact_names_all)
    print(('All contacts have coordinates {}'.format(coords_all)))

    fake_coords = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print(('Setting contact {} coordinates to {}'.format(contact_names1[0], fake_coords[0])))
    loc.set_contact_coordinate('ct_voxel', contact_names1[0], fake_coords[0] )
    coords1 = loc.get_contact_coordinate('ct_voxel', contact_names1[0])
    print(('Contact {} now has coordinates {}'.format(contact_names1[0], coords1)))
    if (fake_coords[0] != coords1).any():
        raise Exception("Coordinates do not match")

    print(('Setting contacts {} coordinates to {}'.format(contact_names1[1:4], fake_coords[1:4])))

    loc.set_contact_coordinates('ct_voxel', contact_names1[1:4], fake_coords[1:4])
    coords_multi = loc.get_contact_coordinates('ct_voxel', contact_names1[1:4])
    print(('Contacts {} now have coordinates {}'.format(contact_names1[1:4], coords_multi)))
    if (fake_coords[1:4] != coords_multi).any():
        raise Exception("Coordinates do not match")

    empty_label = loc.get_contact_label('whole_brain', contact_names1[0])
    if empty_label is not None:
        raise Exception("Label not None before it is set")
    print(("Contact {} has no label for atlas {}".format(contact_names1[0], 'whole_brain')))
    print(("Setting contact {} atlas {} label to {}".format(contact_names1[0], 'whole_brain', 'in_the_brain')))
    loc.set_contact_label('whole_brain', contact_names1[0], 'in_the_brain')
    label = loc.get_contact_label('whole_brain', contact_names1[0])
    print(("Contact {} atlas {} label is now {}".format(contact_names1[0], 'whole_brain', label)))
    if (label != 'in_the_brain'):
        raise Exception("Label not properly set")
    labels_multi = loc.get_contact_labels('whole_brain', contact_names1[0:3])
    print(("Contacts {} have labels {} for atlas {}".format(contact_names1[0:3], labels_multi, 'whole_brain')))
    new_labels = ['on_the_brain', 'off_the_brain']
    print(("setting contacts {} atlas {} labels to {}".format(contact_names1[1:3], 'whole_brain', new_labels)))
    loc.set_contact_labels('whole_brain', contact_names1[1:3], new_labels)
    labels_multi = loc.get_contact_labels('whole_brain', contact_names1[0:3])
    print(("Contacts {} now have atlas {} labels {}".format(contact_names1[0:3], 'whole_brain', labels_multi)))

    info = loc.get_contact_info('displacement', contact_names1[0])
    print(("Contact {} has displacement {}".format(contact_names1[0], info)))
    loc.set_contact_info('displacement', contact_names1[0], 'TOO MUCH!')
    info = loc.get_contact_info('displacement', contact_names1[0])
    print(("Contact {} has displacement {}".format(contact_names1[0], info)))


    info = loc.get_contact_infos('displacement', contact_names1[0:2])
    print(("Contacts {} has displacement {}".format(contact_names1[0:2], info)))
    loc.set_contact_infos('displacement', contact_names1[0:2], ('TOO LITTLE!', 'A LOT!'))
    info = loc.get_contact_infos('displacement', contact_names1[0:2])
    print(("Contact {} has displacement {}".format(contact_names1[0:2], info)))


    pairs_1 = loc.get_pairs(leads[0])
    print(("Lead {} has pairs {}".format(leads[0], pairs_1)))
    pairs_all = loc.get_pairs()
    print(("All pairs in localization are {}".format(pairs_all)))

    coords1 = loc.get_pair_coordinate('ct_voxel', pairs_1[0])
    print(("Pair {} has coordinates {}".format(pairs_1[0], coords1)))
    coords_multi = loc.get_pair_coordinates('ct_voxel', pairs_1[0:3])
    print(("Pairs {} have coordinates {}".format(pairs_1[0:3], coords_multi)))

    empty_label = loc.get_pair_label('whole_brain', pairs_all[0])
    if empty_label is not None:
        raise Exception("Label is not None before it is set")
    print(("Pair {} has no label for atlas {}".format(pairs_all[0], "whole_brain")))
    print(("Setting pairs {} atlas {} label to {}".format(pairs_all[0], 'whole_brain', 'by_the_brain')))
    loc.set_pair_label('whole_brain', pairs_all[0], 'by_the_brain')
    label = loc.get_pair_label('whole_brain', pairs_all[0])
    print(("Pair {} atlas {} label now set to {}".format(pairs_all[0], 'whole_brain', label)))
    new_labels = ['near_the_brain', 'under_the_brain']
    print(("Setting pairs {} atlas {} labels to {}".format(pairs_all[1:3], 'whole_brain', new_labels)))
    loc.set_pair_labels('whole_brain', pairs_all[1:3], new_labels)
    labels = loc.get_pair_labels('whole_brain', pairs_all[0:4])
    print(("Pairs {} now have atlas {} labels {}".format(pairs_all[0:4], 'whole_brain', labels)))
        
    info = loc.get_pair_info('displacement', pairs_all[0])
    print(("Pair {} has displacement {}".format(pairs_all[0], info)))
    loc.set_pair_info('displacement', pairs_all[0], 'TOO MUCH!')
    info = loc.get_pair_info('displacement', pairs_all[0])
    print(("Pair {} has displacement {}".format(pairs_all[0], info)))


    info = loc.get_pair_infos('displacement', pairs_all[0:2])
    print(("Pairs {} has displacement {}".format(pairs_all[0:2], info)))
    loc.set_pair_infos('displacement', pairs_all[0:2], ('TOO LITTLE!', 'A LOT!'))
    info = loc.get_pair_infos('displacement', pairs_all[0:2])
    print(("Pairs {} has displacement {}".format(pairs_all[0:2], info)))

    
    loc.to_json("sample_voxel_coordinates_end.json")
