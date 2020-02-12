import math
import configparser
import os.path
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy import constants
from dataclasses import dataclass
from tqdm import tqdm

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

        if os.path.isfile('{base}/{config}/config.ini'.format(base=os.getcwd(), config=self.config)):
            self.config_parse = configparser.ConfigParser()
            self.config_parse.read('{base}/{config}/config.ini'.format(base=os.getcwd(), config=self.config))

            self.kinematics = PyStruct(self.config_parse['event']['particle'],
                                       float(self.config_parse['event']['energy']),
                                       float(self.config_parse['event']['momentum']),
                                       float(self.config_parse['event']['theta']),
                                       float(self.config_parse['event']['phi']))
        else:
            print("Failed to find configuration file. Exiting.")
            exit(1)

    def calculate_cross_section(self, theta: 'np.array')->'iterator':
        self.kinematics.theta = 0
        for beta in tqdm(theta):
            self.cross_section = math.pow(beta, -2)*math.log( (2*math.pow(constants.speed_of_light*beta, 2))/(1-math.pow(beta, 2)) - math.pow(beta, 2))
            yield self.cross_section

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

    theta = np.array([random.uniform(0.10, 0.99) for i in range(1000)])
    cs = np.fromiter(femto.calculate_cross_section(theta), dtype=float, count = theta.size)

    fig, ax = plt.subplots()
    ax.scatter(theta, cs)
    plt.show()
