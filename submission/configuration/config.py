import os
import yaml
import matplotlib
import argparse
import copy
from ..exc import ConfigurationError

MPL_BACKEND = matplotlib.get_backend()


def yml_join(loader, node):
    return os.path.join(*[str(i) for i in loader.construct_sequence(node)])
yaml.add_constructor('!join', yml_join)


class ConfigOption(object):

    def __init__(self, options):
        self.options = options
        for k, v in options.items():
            setattr(self, k, v)

    def set(self, k, v):
        if k not in self.options:
            raise ConfigurationError("Attempted to set invalid option {}. "
                                     "Valid options are {}".format(k, self.options.keys()))
        self.options[k] = v
        setattr(self, k, v)

    def get(self, k, default=None):
        return self.options.get(k, default)

    def __str__(self):
        return repr(self) + '(' + ', '.join('{}={}'.format(k, getattr(self, k)) for k in self.options.keys() ) + ')'

    def __contains__(self, item):
        return item in self.options


class Configuration(object):

    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yml')

    def __init__(self, config_file=None):
        if config_file is None:
            self.config_file = self.DEFAULT_CONFIG_FILE

        self.config_dict = {}
        self.options = {}
        self.parser = argparse.ArgumentParser()

        self.load_config(self.config_file)

    def load_config(self, config_file):
        self.config_file = config_file
        self.config_dict = yaml.load(open(self.config_file))

        options = self.config_dict['options']

        for option in options:
            arg = option.get('arg')
            if arg is not None:
                self.add_argument(option)

            if 'options' not in option:
                self.options[option['dest']] = option.get('default', False)
            else:
                subfield = ConfigOption(option['options'])
                self.options[option['dest']] = subfield

    def add_argument(self, option):
        cpy = copy.copy(option)
        name = '--{arg}'.format(**option)
        del cpy['arg']

        if 'action' not in cpy:
            cpy['action'] = 'store_true'

        if cpy['action'] == 'store_true':
            cpy['default'] = cpy.get('default', False)

        if 'options' in cpy:
            cpy['help'] = cpy.get('help', '')
            cpy['help'] += ' Available options are: {}'.format(', '.join(cpy['options']))
            del cpy['options']

        self.parser.add_argument(name, **cpy)

    def parse_args(self, *args, **kwargs):
        parsed = self.parser.parse_args(*args, **kwargs)

        for k, v in self.options.items():
            new_v = getattr(parsed, k)
            if isinstance(v, ConfigOption):
                if new_v is None:
                    continue
                for val in new_v:
                    for val_item in val.split(':'):
                        v.set(*val_item.split('='))
            else:
                self.options[k] = new_v


    def __str__(self):
        return repr(self) + '(' + ', '.join('{}={}'.format(k,v) for k,v in self.options.items()) + ')'

    def __getattr__(self, item):
        try:
            return vars(self)[item]
        except KeyError:
            return self.options[item]

if __name__ == '__main__':
    config = Configuration()
    config.parse_args(['--path','rhino_root=abc'])
    print(config)
