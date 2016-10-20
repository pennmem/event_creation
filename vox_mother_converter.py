import re
import numpy as np
import os
from collections import defaultdict
from config import RHINO_ROOT

class Contact(object):

    def __init__(self, contact_name=None, contact_num=None, coords=None, type=None, size=None):
        self.name = contact_name
        self.coords = coords
        self.num = contact_num
        self.type = type
        self.grid_size = size
        self.grid_group = None
        self.jack_num = None
        self.grid_loc = None

def read_mother(mother_file):
    leads = defaultdict(lambda: defaultdict(Contact))
    for line in open(mother_file):
        split_line = line.strip().split()
        contact_name = split_line[0]
        coords = [int(x) for x in split_line[1:4]]
        type = split_line[4]
        size = tuple(int(x) for x in split_line[5:7])
        match = re.match(r'(.+?)(\d+$)', contact_name)
        lead_name = match.group(1)
        contact_num = int(match.group(2))
        contact = Contact(contact_name, contact_num, coords, type, size)
        leads[lead_name][contact_num] = contact
    return leads

def add_jacksheet(leads, jacksheet_file):
    skipped_leads = []
    for line in open(jacksheet_file):
        split_line = line.split()
        jack_num = split_line[0]
        contact_name = split_line[1]
        match = re.match(r'(.+?)(\d+$)', contact_name)
        if not match:
            print 'Warning: cannot parse contact {}. Skipping'.format(contact_name)
            continue
        lead_name = match.group(1)
        contact_num = int(match.group(2))
        if lead_name not in leads:
            if lead_name not in skipped_leads:
                print('Warning: skipping lead {}'.format(lead_name))
                skipped_leads.append(lead_name)
            continue
        if contact_num not in leads[lead_name]:
            print('Error: neural lead {} missing contact {}'.format(lead_name, contact_num))
            continue
        leads[lead_name][contact_num].jack_num = jack_num

def add_grid_loc_old(leads):
    for lead_name, lead in leads.items():
        types = set(contact.type for contact in lead.values())
        if len(types) > 1:
            raise Exception("Cannot convert lead with multiple types")
        type = types.pop()
        if type != 'G':
            for contact in lead.values():
                contact.grid_loc = (1, contact.num)
        grid_sizes = set(contact.grid_size for contact in lead.values())
        for grid_size in grid_sizes:
            subgrid = {num: contact for num, contact in lead.items() if contact.grid_size == grid_size}
            
            if len(subgrid) > grid_size[0] * grid_size[1]:
                print 'ERROR: Length of lead {} ({}) is larger than product of its dimensions {}'.format(lead_name, len(subgrid), grid_size)
            if len(subgrid) == grid_size[0] * grid_size[1]:
                start = min(subgrid.keys())
            else:
                nums = sorted(subgrid.keys())
                min_num = min(nums)
                previous_contacts = [contact for contact in lead.values() if contact.grid_size != grid_size and contact.num < min_num]
                if previous_contacts:
                    start = max(previous_contacts) + 1
                else:
                    start = 1
            for num, contact in subgrid.items():
                if (num-start)/grid_size[0] >= grid_size[1]:
                    print 'The number for contact {} is too high to fit inside the grid'.format(contact.name)
                    continue
                contact.grid_loc = ((num-start) % grid_size[0], (num-start) / grid_size[0])

def add_grid_loc(leads):
   for lead_name, lead in leads.items():
        types = set(contact.type for contact in lead.values())
        if len(types) > 1:
            raise Exception("Cannot convert lead with multiple types")
        type = types.pop()
        if type != 'G':
            for contact in lead.values():
                contact.grid_loc = (1, contact.num)

        sorted_contact_nums = sorted(lead.keys())
        previous_grid_size = lead[sorted_contact_nums[0]].grid_size
        group = 0
        start_num = 1
        for num in sorted_contact_nums:
            contact = lead[num]
            if contact.grid_size != previous_grid_size:
                group += 1
                start_num = num 
            if num - start_num >= contact.grid_size[0] * contact.grid_size[1]:
                group += 1
                start_num = num

            contact.grid_loc = ((num-start_num) % contact.grid_size[0], (num-start_num)/contact.grid_size[0])
            contact.grid_group = group

def build_leads(files):
    leads = read_mother(files['leads'])
    add_jacksheet(leads, files['jacksheet'])
    add_grid_loc(leads)
    return leads

def file_locations(subject):
    files = dict(
        leads=os.path.join(RHINO_ROOT, 'data', 'eeg', subject, 'tal', 'VOX_coords_mother.txt'),
        jacksheet=os.path.join(RHINO_ROOT, 'data', 'eeg', subject, 'docs', 'jacksheet.txt')
    )
    return files

if __name__ == '__main__':
    build_leads(default_file_locations('R1009W'))
