import math
import configparser
import os.path
import numpy as np
import pandas as pd

from scipy import constants
from dataclasses import dataclass
from tqdm import tqdm

from .pymetric import pymetric

class FemtoGen:

    def __init__(self, *args, **kawgs):
        self.config = 'config'
        self.kinematics = 0
        self.GAMMA = 0
        self.tau = 0
        self.data_file = ''
        self.proton_mass = 0.93828

        self.Ge = 0
        self.Gm = 0
        self.F2 = 0
        self.F1 = 0

        self.Q = 0
        self.nu = 0
        self.gamma = 0
        self.tau = 0
        self.y = 0
        self.q_0 = 0
        self.kp_0 = 0
        self.eps = 0
        self.xi = 0
        self.t_0 = 0

        # Trig functions
        self.cosl = 0
        self.sinl = 0
        self.coslp = 0
        self.sinlp = 0
        self.cost = 0
        self.sint = 0

        # Four-vectors
        self.q = 0
        self.qp = 0
        self.p = 0
        self.delta = 0
        self.pp = 0
        self.P = 0
        self.k = 0
        self.kp = 0

        # Contractions
        self.kk = 0
        self.PP = 0
        self.k_qp = 0

        self.kP = 0
        self.k_kp = 0
        self.kp_P = 0
        self.kp_qp = 0

        self.P_qp = 0
        self.kd = 0
        self.kp_d = 0
        self.qpd = 0

        self.kk_t = 0
        self.kqp_t = 0
        self.kkp_t = 0
        self.kpqp_t = 0
        self.kP_t = 0
        self.kpP_t = 0
        self.qpP_t = 0
        self.kd_t = 0
        self.kpd_t = 0
        self.qpd_t = 0

        self.s = 0

        self.GAMMA = 0

        self.D_plus = 0
        self.D_minus = 0

        self.xbj = []
        self.t = []
        self.Q2 = []
        self.k_0 = []
        self.ReH = []
        self.ImE = []
        self.ImH = []
        self.ReE = []
        self.ReHt = []
        self.ImHt = []
        self.ReEt = []
        self.ImEt = []

        self.ke = 8.985551e9  # N m^2 C-2 Coulomb's constant
        self.metric = pymetric.Metric()
        self.metric.set_minkowski_metric()

        self.cross_section = {'bh': 0,
                              'dvcs': 0,
                              'int': 0,
                              'full': 0}

        if "config" in kawgs:
            self.config = kawgs['config']
        else:
            pass

    def update_elastic_form_factors(self, t: float) -> 'float, float, float, float':
        '''

        :param t:
        :return: The Fermi, Dirc, Electric, Magnetic form factors
        '''
        self.Ge = 1 / math.pow(1 + t / 0.710649, 2)
        self.Gm = 2.792847337 * self.Ge
        self.F2 = (self.Gm - self.Ge) / (1 + (t / (4 * math.pow(0.938, 2))))
        self.F1 = self.Gm - self.F2

        return self.F1, self.F2, self.Ge, self.Gm

    def read_config(self, config: str):
        """
        Read configuration file from standard {base}/config directory.
        """

        if os.path.isfile('{base}/{config}/config.ini'.format(base=os.getcwd(), config=self.config)):
            self.config_parse = configparser.ConfigParser()
            self.config_parse.read('{base}/{config}/config.ini'.format(base=os.getcwd(), config=self.config))

            self.kinematics = PyStruct(float(self.config_parse['event']['xbj']),
                                       float(self.config_parse['event']['t']),
                                       float(self.config_parse['event']['Q2']),
                                       float(self.config_parse['event']['k_0']))
        else:
            print("Failed to find configuration file. Exiting.")
            exit(1)

    def calculate_kinematics(self, i: int, phi: float):
        '''

        :param i:
        :param phi:
        :return:
        '''

        self.Q = math.sqrt(self.Q2[i])
        self.nu = self.Q2[i] / (2 * self.proton_mass * self.xbj[i])
        self.gamma = self.Q / self.nu
        self.tau = -self.t[i] / (4 * math.pow(self.proton_mass, 2))
        self.y = self.Q / (self.gamma * self.k_0[i])
        self.q_0 = (self.Q / self.gamma) * (1 + self.xbj[i] * self.t[i] / self.Q2[i])
        self.kp_0 = self.k_0[i] * (1 - self.y)
        self.eps = (1 - self.y - math.pow(0.5 * self.y * self.gamma, 2)) / (
                1 - self.y + 0.5 * math.pow(self.gamma, 2) + math.pow(0.5 * self.y * self.gamma, 2))
        self.xi = self.xbj[i] * (
                (1 + (self.t[i] / (2 * self.Q2[i]))) / (2 - self.xbj[i] + ((self.xbj[i] * self.t[i]) / self.Q2[i])))
        self.t_0 = self.Q2[i] * (1 - math.sqrt(1 + math.pow(self.gamma, 2))
                                 + 0.5 * math.pow(self.gamma, 2)) / (self.xbj[i] * (
                1 - math.sqrt(1 + math.pow(self.gamma, 2)) + math.pow(self.gamma, 2) / (2 * self.xbj[i])))

        # Trig functions
        self.cosl = -(1 / math.sqrt(1 + math.pow(self.gamma, 2))) * (1 + 0.5 * self.y * math.pow(self.gamma, 2))
        self.sinl = (self.gamma / math.sqrt(1 + math.pow(self.gamma, 2))) * math.sqrt(
            1 - self.y - math.pow(0.5 * self.gamma * self.y, 2))
        self.coslp = (self.cosl + self.y * math.sqrt(1 + math.pow(self.gamma, 2))) / (1 - self.y)
        self.sinlp = self.sinl / (1 - self.y)
        self.cost = -(1 / math.sqrt(1 + math.pow(self.gamma, 2))) * (1 + (.5 * math.pow(self.gamma, 2)) * (
                (1 + (self.t[i] / self.Q2[i])) / (1 + ((self.xbj[i] * self.t[i]) / (self.Q2[i])))))
        self.sint = math.sqrt(1 - math.pow(self.cost, 2))

        # Four-vectors
        self.q = np.array([self.nu, 0, 0, -self.nu * math.sqrt(1 + math.pow(self.gamma, 2))])
        self.qp = self.q_0 * np.array([1, self.sint * math.cos(phi), self.sint * math.sin(phi), self.cost])
        self.p = np.array([self.proton_mass, 0, 0, 0])
        self.delta = self.q - self.qp
        self.pp = self.p + self.delta
        self.P = 0.5 * (self.p + self.pp)
        self.k = self.k_0[i] * np.array([1, self.sinl, 0, self.cosl])
        self.kp = self.kp_0 * np.array([1, self.sinlp, 0, self.coslp])

        # Contractions
        self.kk = self.metric.contract(self.k, self.k)
        self.PP = self.metric.contract(self.p, self.p)
        self.k_qp = self.metric.contract(self.k, self.qp)

        self.kP = self.metric.contract(self.k, self.P)
        self.k_kp = self.metric.contract(self.k, self.kp)
        self.kp_P = self.metric.contract(self.kp, self.P)
        self.kp_qp = self.metric.contract(self.kp, self.qp)

        self.P_qp = self.metric.contract(self.P, self.qp)
        self.kd = self.metric.contract(self.k, self.delta)
        self.kp_d = self.metric.contract(self.kp, self.delta)
        self.qpd = self.metric.contract(self.qp, self.delta)

        self.kk_t = self.metric.contract(self.k, self.k, type='transverse')
        self.kqp_t = self.metric.contract(self.k, self.qp, type='transverse')
        self.kkp_t = self.metric.contract(self.k, self.kp, type='transverse')
        self.kpqp_t = self.metric.contract(self.kp, self.qp, type='transverse')
        self.kP_t = self.metric.contract(self.k, self.P, type='transverse')
        self.kpP_t = self.metric.contract(self.kp, self.P, type='transverse')
        self.qpP_t = self.metric.contract(self.qp, self.P, type='transverse')
        self.kd_t = self.metric.contract(self.k, self.delta, type='transverse')
        self.kpd_t = self.metric.contract(self.kp, self.delta, type='transverse')
        self.qpd_t = self.metric.contract(self.qp, self.delta, type='transverse')

        self.s = self.kk + self.metric.contract(self.p, self.p) + 2 * self.metric.contract(self.k, self.p)

        self.GAMMA = math.pow(constants.alpha, 3) / (
                16 * self.xbj[i] * math.pow(constants.pi, 2) * math.pow((self.s - math.pow(self.proton_mass, 2)),
                                                                        2) * math.sqrt(1 + math.pow(self.gamma, 2)))

        self.D_plus = (1 / (2 * self.kp_qp)) - (1 / (2 * self.k_qp))
        self.D_minus = -(1 / (2 * self.kp_qp)) - (1 / (2 * self.k_qp))

    def calculate_interference(self, i, phi: 'float'):
        '''

        :param i:
        :param phi:
        :return:
        '''
        self.calculate_kinematics(i, phi)
        A_UU = -4 * math.cos(phi) * (self.D_plus * ((self.kqp_t - 2 * self.kk_t - 2 * self.k_qp) * self.kp_P
                                                    + (
                                                            2 * self.kp_qp - 2 * self.kkp_t - self.kpqp_t) * self.kP + self.kp_qp * self.kP_t
                                                    + self.k_qp * self.kpP_t - 2 * self.k_kp * self.kP_t)
                                     - self.D_minus * ((
                                                               2 * self.k_kp - self.kpqp_t - self.kkp_t) * self.P_qp + 2 * self.k_kp * self.qpP_t
                                                       - self.kp_qp * self.kP_t - self.k_qp * self.kpP_t))

        B_UU = -2 * self.xi * math.cos(phi) * (self.D_plus * (
                (self.kqp_t - 2 * self.kk_t - 2 * self.k_qp) * self.kp_d + (
                    2 * self.kp_qp - 2 * self.kkp_t - self.kpqp_t) * self.kd
                + self.kp_qp * self.kd_t + self.k_qp * self.kpd_t - 2 * self.k_kp * self.kd_t)
                                               - self.D_minus * ((
                                                                         2 * self.k_kp - self.kpqp_t - self.kkp_t) * self.qpd + 2 * self.k_kp * self.qpd_t
                                                                 - self.kp_qp * self.kd_t - self.k_qp * self.kpd_t))
        C_UU = -2 * math.cos(phi) * (
                self.D_plus * (2 * self.k_kp * self.kd_t - self.kp_qp * self.kd_t - self.k_qp * self.kpd_t
                               + 4 * self.xi * self.k_kp * self.kP_t - 2 * self.xi * self.kp_qp * self.kP_t - 2 * self.xi * self.k_qp * self.kpP_t)
                - self.D_minus * (
                        self.k_kp * self.qpd_t - self.kp_qp * self.kd_t - self.k_qp * self.kpd_t + 2 * self.xi * self.k_kp * self.qpP_t
                        - 2 * self.xi * self.kp_qp * self.kpP_t - 2 * self.xi * self.k_qp * self.kpP_t))

        return A_UU, B_UU, C_UU

    def calculate_bethe_heitler(self, i, phi):
        '''

        :param i:
        :param phi:
        :return:
        '''
        self.calculate_kinematics(i, phi)
        const = 8 * math.pow(self.proton_mass, 2) / (self.t[i] * self.t[i] * self.k_qp * self.kp_qp)
        A_UU = -const * (4 * self.tau * (math.pow(self.kP, 2) + math.pow(self.kp_P, 2)) - (self.tau + 1) * (
                math.pow(self.kd, 2) + math.pow(self.kp_d, 2)))
        B_UU = -2 * const * (math.pow(self.kd, 2) + math.pow(self.kp_d, 2))

        return A_UU, B_UU, 0

    def calculate_dvcs(self, i, phi):
        '''

        :param i:
        :param phi:
        :return:
        '''
        self.calculate_kinematics(i, phi)

        A = 4 * (1 - math.pow(self.xi, 2)) * (
                math.pow(self.ReH[i], 2) + math.pow(self.ImH[i], 2) + math.pow(self.ReHt[i], 2) + math.pow(self.ImHt[i],
                                                                                                           2))
        B = ((self.t_0 - self.t[i]) / 2 * math.pow(self.proton_mass, 2)) * (
                math.pow(self.ReE[i], 2) + math.pow(self.ImE[i], 2) + math.pow(self.xi, 2) * (
                    math.pow(self.ReEt[i], 2) + math.pow(self.ImEt[i], 2)))
        C = ((2 * math.pow(self.xi, 2)) / (1 - math.pow(self.xi, 2))) * (
                self.ReE[i] * self.ReH[i] + self.ImE[i] * self.ImH[i] + self.ReEt[i] * self.ReHt[i] + self.ImEt[i] *
                self.ImHt[i])

        F_UUT = (A + B - C)

        return F_UUT, 0, 0

    def generate_cross_section(self, phi: 'np.array', type='full', error=None) -> 'iterator':
        """
           Generator for the DVCS cross section as a function of phi
        """

        if type not in self.cross_section:
            print('Invalid cross-section choice.')
            yield 0

        self.update_elastic_form_factors(-self.t[0])

        for p in tqdm(phi):
            # DVCS Term
            conversion = math.pow(.197326, 2) * 1e7 * (2 * math.pi)
            ADVCS, _, _ = self.calculate_dvcs(0, p)

            A_DVCS = ADVCS * conversion * (self.GAMMA / (self.Q2[0] * (1 - self.eps)))

            self.cross_section['dvcs'] = (A_DVCS)

            # BH Term
            ABH, BBH, _ = self.calculate_bethe_heitler(0, p)
            A_BH = -ABH * conversion * self.GAMMA * (math.pow(self.F1, 2) + self.tau * math.pow(self.F2, 2))
            B_BH = -BBH * conversion * self.GAMMA * self.tau * math.pow(self.Gm, 2)

            self.cross_section['bh'] = (A_BH + B_BH)

            # Interference Term
            A, B, C = self.calculate_interference(0, p)
            A_term = -A * conversion * (self.GAMMA / (-self.t[0] * self.Q2[0])) * (
                    self.F1 * self.ReH[0] + self.tau * self.F2 * self.ReE[0])
            B_term = -B * conversion * (self.GAMMA / (-self.t[0] * self.Q2[0])) * (self.F1 + self.F2) * (
                    self.ReH[0] + self.ReE[0])
            C_term = -C * conversion * (self.GAMMA / (-self.t[0] * self.Q2[0])) * (self.F1 + self.F2) * self.ReHt[0]

            self.cross_section['int'] = (A_term + B_term + C_term)

            self.cross_section['full'] = self.cross_section['bh'] + self.cross_section['dvcs'] + self.cross_section[
                'int']

            yield self.cross_section[type] + next(error)

    def error_generator(self, mean=0.0, stdev=1.0, systematic=0.0):
        while True:
            yield np.random.normal(mean, stdev) + systematic

    def read_data_file(self, file: str):
        self.data_file = file
        df = pd.read_csv(file)
        print(df.head())

        self.xbj = df['xbj'].to_numpy()
        self.t = df['t'].to_numpy()
        self.Q2 = df['Q2'].to_numpy()
        self.k_0 = df['k_0'].to_numpy()
        self.ReH = df['ReH'].to_numpy()
        self.ImH = df['ImH'].to_numpy()
        self.ReE = df['ReE'].to_numpy()
        self.ImE = df['ImE'].to_numpy()
        self.ReHt = df['ReHt'].to_numpy()
        self.ImHt = df['ImHt'].to_numpy()
        self.ReEt = df['ReEt'].to_numpy()
        self.ImEt = df['ImEt'].to_numpy()

@dataclass
class PyStruct:
    xbj: float = 0.0
    t: float = 0.0
    Q2: float = 0.0
    k_0: float = 0.0