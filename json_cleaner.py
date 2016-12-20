"""
A simple library to format JSON strings in the format that I wanted them.
Removes linebreaks that do not correspond to the start of a JSON object
"""

import re
import json

def clean_dump(object, fd, *args, **kwargs):
    """
    Dump JSON representation of an object to a file
    >>> clean_dump({'hello': 'world'}, 
                   open('file_out.json', 'w'),
                   indent=2, sort_keys=True)
    :param object: The object to dump to the json file
    :param fd: File descriptor to which the json should be dumped
    :param *args: args to supply to json.dumps
    :param **kwargs: keyword arguments to supply to json.dumps
    """
    fd.write(clean_dumps(object, *args, **kwargs))

def clean_dumps(object, *args, **kwargs):
    """
    Dump JSON representation of an object to a string
    >>> clean_dump({'hello': 'world'}, indent=2)
    :param object: The object to dump to json string
    :param *args: args to supply to json.dumps
    :param **kwargs: keyword arguments to supply to json.dumps
    :return: The string representation
    """
    return clean_json(json.dumps(object, *args, **kwargs))

def clean_json(json_string):
    """
    Cleans a json string, removing linebreaks that do not correspond
    to the start of a JSON object (as oppsed to a list member)
    """
    new_lines = []
    in_list = False
    for line in json_string.split('\n'):
        if ']' in line:
            in_list = False
            new_lines[-1] += line.strip()
        elif '[' in line:
            in_list = True
            new_lines.append(line + ' ')
        elif in_list and line.strip()[0].isdigit():
            new_lines[-1] += line.strip() + ' '
        else:
            new_lines.append(line)  
    return '\n'.join(new_lines)

