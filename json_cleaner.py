import re
import json

def clean_dump(object, fd, *args, **kwargs):
    fd.write(clean_dumps(object, *args, **kwargs))

def clean_dumps(object, *args, **kwargs):
    return clean_json(json.dumps(object, *args, **kwargs))

def clean_json(json_string):
    new_lines = []
    for line in json_string.split('\n'):
        if not ('{' in line ):
            new_lines[-1] += line.strip()
        #if re.match('\s+((\d+)|]),?\s*$', line):
        #    new_lines[-1] += line.strip()
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def test_clean_json():
    x = open('voxel_coordinates.json').read()
    print clean_json(x)
