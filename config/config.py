
import os
import yaml
import matplotlib
import argparse
import copy

MPL_BACKEND = matplotlib.get_backend()

def yml_join(loader, node):
    return os.path.join(*[str(i) for i in loader.construct_sequence(node)])
yaml.add_constructor('!join', yml_join)

class ConfigurationException(Exception):
    pass

class ConfigOption(object):

    def __str__(self):
        return repr(self) + '(' + ', '.join('{}={}'.format(k,v) for k,v in vars(self).items()) + ')'

class Configuration(object):

    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yml')
    CONFIG_HANDLERS = {}

    def show_plot_handler(self, value):
        if value:
            matplotlib.use(MPL_BACKEND)
        else:
            matplotlib.use('agg')

    CONFIG_HANDLERS['show_plot'] = show_plot_handler

    def __init__(self, config_file=None):
        if config_file is None:
            self.config_file = self.DEFAULT_CONFIG_FILE

        self.config_dict = {}

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
                setattr(self, option['dest'], option.get('default', False))
            else:
                subfield = ConfigOption()
                setattr(self, option['dest'], subfield)
                for k, v in option['options'].items():
                    setattr(subfield, k, v)

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
        self.parser.parse_args(*args, **kwargs)

    def __str__(self):
        return repr(self) + '(' + ', '.join('{}={}'.format(k,v) for k,v in vars(self).items() if k != 'config_dict') + ')'

if __name__ == '__main__':
    config = Configuration()
    config.parse_args()
    print config