# Utility to extract all file names (including nested) from ephys_inputs.yml
import yaml
import os

def yml_join(loader, node):
    return os.path.join(*[str(i) for i in loader.construct_sequence(node)])

yaml.add_constructor('!join', yml_join)

def extract_file_names(yml_path):
    with open(yml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    files = data.get('files', [])
    names = set()
    def recurse(files):
        for entry in files:
            name = entry.get('name')
            if name:
                names.add(name)
            if 'files' in entry:
                recurse(entry['files'])
    recurse(files)
    return names

if __name__ == '__main__':
    yml_path = os.path.join(os.path.dirname(__file__), 'transfer_inputs', 'ephys_inputs.yml')
    names = extract_file_names(yml_path)
    print(sorted(names))
