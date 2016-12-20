class Contacts(object):

    def __init__(self, voxel_coordinates_json):
        pass

    def get_lead_names(self):
        """ Returns list of strings of lead names """
        pass

    def get_contact_type(self, contact_name):
        """
        Returns the full dictionary for a given contact
        (includes type of electrode, "s", "d")
        """

    def get_contact_names(self, lead=None):
        """ 
        Returns list of strings of contacts
        for each lead if provided, otherwise all contacts
        """
        pass

    def get_contact_coordinates(self, coord_type, contacts):
        """
        coord_type: one of "fs", "t1_mri", "ct_voxels", ...
        contacts: list of strings of contact names
        returns: np.array of coordinates for each contact
        """
        pass
    
    def add_contact_coordinate(self, coordinates, coord_type, contacts):
        """
        Sets the coordinates for a given contact and coordinate type
        """
        pass

    def add_contact_label(self, atlas_name, contact, label):
        """
        Sets the label for a given contact
        """
        pass

    def get_pairs(self, lead=None):
        """
        Returns a list of 2-tuples of strings with
        contacts names that form a pair
        """
        pass
    
    def get_pair_coordinates(self, coord_type, pairs):
        """
        coord_type: one of "fs", "t1_mri", "ct_voxels", ...
        pairs: Nx2 list of contact names (output of get_pairs)
        returns: np.array of coordinates for each pair
        """
        pass

    def add_pair_label(self, atlas_name, pair, label):
        """
        Sets the label for a given pair
        """
        pass
    
