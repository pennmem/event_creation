
import json
import os
import yaml



class ConfigReader(object):

    JSON_FILES = {}

    def __init__(self, filename):
        self.filename = filename


    @classmethod
    def load_json_input(cls, json_file):
        if json_file not in cls.JSON_FILES:
            cls.JSON_FILES[json_file] = json.load(open(json_file))
        return cls.JSON_FILES[json_file]

    @classmethod
    def load_yaml_input(cls, yaml_file):
        return yaml.load(open(yaml_file))