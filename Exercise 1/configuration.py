# -*- coding: utf-8 -*-
import configparser
import os

TRUE_VALUE = ['True', 'true', 'TRUE', '1']


class Configuration:

    def __init__(self, config_file: str = os.path.join(os.getcwd(), '.config')):
        assert isinstance(config_file, str), "the config_file must be a string"

        # parse configuration
        # first parse the .config file and afterwards the commandline arguments
        self._parse_ini_file(config_file)

    def _parse_ini_file(self, config_file: str):
        # generate empty dictionary
        self._config = dict()

        # start read config file
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf8')

        for key in config:
            sub_config = config[key]

            for sub_key in sub_config:
                name = key.upper() + '_' + sub_key.upper()
                value = sub_config[sub_key]

                self._config[name] = value if (value != '') else None

    @property
    def communities_data(self):
        return self['DATASET_COMMUNITIES_DATA_PATH']

    @property
    def communities_figures(self):
        return self['DATASET_COMMUNITIES_FIGURES_PATH']

    @property
    def traffic_data(self):
        return self['DATASET_TRAFFIC_DATA_PATH']

    @property
    def traffic_figures(self):
        return self['DATASET_TRAFFIC_FIGURES_PATH']

    @property
    def real_estate_data(self):
        return self['DATASET_REAL_ESTATE_DATA_PATH']

    @property
    def real_estate_figures(self):
        return self['DATASET_REAL_ESTATE_FIGURES_PATH']


    def __getitem__(self, item):
        # check if the item exists and if not return None
        # not the Pythonic style of coding
        if item in self._config:
            return self._config[item]
        else:
            return None


# singleton
#print(os.getcwd())
#print(os.path.join(os.getcwd(), '.config'))
default = Configuration()
