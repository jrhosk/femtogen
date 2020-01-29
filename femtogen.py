from scipy import constants
import math
from dataclasses import dataclass
import configparser
import os.path


class FemtoGen:
    def __init__(self, *args, **kawgs):
        self.config = 'config'
        self.kinematics = 0
        self.cross_section = 0

        self.ke = 8.985551e9  # N m^2 C-2 Coulomb's constant

        if "config" in kawgs:
            self.config = kawgs['config']
        else:
            pass

    def read_config(self, config: str):
        """

        """
        if os.path.isfile('{base}/femtogen/{config}/config.ini'.format(base=os.getcwd(), config=self.config)):
            self.config_parse = configparser.ConfigParser()
            self.config_parse.read('{base}/femtogen/{config}/config.ini'.format(base=os.getcwd(), config=self.config))

            self.kinematics = PyStruct(self.config_parse['event']['particle'],
                                       float(self.config_parse['event']['energy']),
                                       float(self.config_parse['event']['momentum']),
                                       float(self.config_parse['event']['theta']),
                                       float(self.config_parse['event']['phi']))
        else:
            print("Failed to find configuration file. Exiting.")
            exit(1)

    def calculate_cross_section(self):
        self.cross_section = 0.5 * constants.pi * math.pow(1, 2) * math.pow(1, 2) * math.pow(constants.alpha,
                                                                                             2) * math.pow(
            (0.197) / (self.kinematics.energy), 2) * (1 / math.pow(1 - math.cos(self.kinematics.theta), 2))
        return self.cross_section


@dataclass
class PyStruct:
    name: str = ''
    energy: float = 0.0
    momentum: float = 0.0
    theta: float = 0.0
    phi: float = 0.0


if __name__ == "__main__":
    femto = FemtoGen()
    femto.read_config('config')
    print(femto.calculate_cross_section())
