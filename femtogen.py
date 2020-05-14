import math
import configparser
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import constants
from dataclasses import dataclass
from tqdm import tqdm

class FemtoGen:
    def __init__(self, *args, **kawgs):
        self.config = 'config'
        self.kinematics = 0
        self.cross_section = 0
        self.GAMMA = 0
        self.tau = 0
        self.data_file = ''
        self.proton_mass = 0.93827208816

        self.Ge = 0
        self.Gm = 0
        self.F2 = 0
        self.F1 = 0


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
        self.metric = self.minkowski_metric(np.identity(4))
        
        if "config" in kawgs:
            self.config = kawgs['config']
        else:
            pass
    def update_elastic_form_factors(self, t: float)->'float, float, float, float':
        self.Ge = 1/math.pow(1 + t/0.710649, 2)
        self.Gm = 2.792847337*self.Ge
        self.F2 = (self.Gm - self.Ge)/(1 + (t/(4*math.pow(0.938, 2))))
        self.F1 = self.Gm - self.F2

        return self.F1, self.F2, self.Ge, self.Gm


    def minkowski_metric(self, metric: 'numpy array')->'numpy array':
        """
           Returns the 4x4 mikowski metric with diagonal of {-1, 1, 1, 1}
        """
        return(np.multiply(metric, np.array([1, -1, -1, -1])))

    
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

    def calculate_kinematics(self, i: int, phi: float)->'float, float, float':
        Q = math.sqrt(self.Q2[i])
        nu = self.Q2[i]/(2*self.proton_mass*self.xbj[i])
        gamma = Q/nu
        self.tau = -self.t[i]/(4*math.pow(self.proton_mass, 2))
        y = Q/(gamma*self.k_0[i])
        q_0 = (Q/gamma)*(1 + self.xbj[i]*self.t[i]/self.Q2[i])
        kp_0 =  self.k_0[i]*(1 - y)
        eps = (1 - y - math.pow(0.5*y*gamma, 2))/(1 - y + 0.5*math.pow(gamma, 2) + math.pow(0.5*y*gamma, 2))
        xi = self.xbj[i]*((1 + (self.t[i]/(2*self.Q2[i])))/(2 - self.xbj[i] + ((self.xbj[i]*self.t[i])/self.Q2[i])))

        # Trig functions
        cosl = -(1/math.sqrt(1 + math.pow(gamma, 2)))*(1 + 0.5*y*math.pow(gamma, 2))
        sinl = (gamma/math.sqrt(1 + math.pow(gamma, 2)))*math.sqrt(1 - y - math.pow(0.5*gamma*y, 2))
        coslp = (cosl + y*math.sqrt(1 + math.pow(gamma, 2)))/(1 - y)
        sinlp = sinl/(1 - y)
        cost = -(1/math.sqrt(1 + math.pow(gamma, 2)))*(1 + (.5*math.pow(gamma,2))*((1+(self.t[i]/self.Q2[i]))/(1 + ((self.xbj[i]*self.t[i])/(self.Q2[i])))))
        sint = math.sqrt(1-math.pow(cost, 2))

        # Four-vectors
        q = np.array([nu, 0, 0, -nu * math.sqrt(1 + math.pow(gamma, 2))])
        qp = q_0 * np.array([1, sint * math.cos(phi), sint * math.sin(phi), cost])
        p = np.array([self.proton_mass, 0, 0, 0])
        delta = q - qp
        pp = p + delta
        P = 0.5*(p + pp)
        k = self.k_0[i]*np.array([1, sinl, 0, cosl])
        kp = kp_0*np.array([1, sinlp, 0, coslp])


        # Contractions
        kk = self.contract(k, k)
        PP = self.contract(p, p)
        k_qp = self.contract(k, qp)

        kP = self.contract(k,P)
        k_kp = self.contract(k,kp)
        kp_P = self.contract(kp,P)
        kp_qp = self.contract(kp,qp)

        P_qp = self.contract(P,qp)
        kd  = self.contract(k, delta)
        kp_d = self.contract(kp, delta)
        qpd = self.contract(qp, delta)

        kk_t = self.contract(k, k, type='transverse')
        kqp_t = self.contract(k, qp, type='transverse')
        kkp_t = self.contract(k, kp, type='transverse')
        kpqp_t = self.contract(kp, qp, type='transverse')
        kP_t = self.contract(k, P, type='transverse')
        kpP_t = self.contract(kp, P, type='transverse')
        qpP_t =  self.contract(qp, P, type='transverse')
        kd_t =  self.contract(k, delta, type='transverse')
        kpd_t = self.contract(kp, delta, type='transverse')
        qpd_t = self.contract(qp, delta, type='transverse')

        s = kk + self.contract(p,p) + 2*self.contract(k,p)

        self.GAMMA = math.pow(constants.alpha, 3)/(16*self.xbj[i]*math.pow(constants.pi, 2)*math.pow((s - math.pow(self.proton_mass, 2)), 2)*math.sqrt(1 + math.pow(gamma, 2)))

        D_plus = (1/(2*kp_qp)) - (1/(2*k_qp))
        D_minus = -(1/(2*kp_qp)) - (1/(2*k_qp))

        A_UU = -4*math.cos(phi)*(D_plus*((kqp_t - 2*kk_t - 2*k_qp)*kp_P + (2*kp_qp - 2*kkp_t - kpqp_t)*kP + kp_qp*kP_t + k_qp*kpP_t - 2*k_kp*kP_t) - D_minus*((2*k_kp - kpqp_t - kkp_t)*P_qp +2*k_kp*qpP_t - kp_qp*kP_t - k_qp*kpP_t))
        B_UU = -2*xi*math.cos(phi)*(D_plus*((kqp_t - 2*kk_t - 2*k_qp)*kp_d + (2*kp_qp - 2*kkp_t - kpqp_t)*kd + kp_qp*kd_t + k_qp*kpd_t - 2*k_kp*kd_t) - D_minus*((2*k_kp - kpqp_t - kkp_t)*qpd + 2*k_kp*qpd_t - kp_qp*kd_t - k_qp*kpd_t))
        C_UU = -2*math.cos(phi)*(D_plus*(2*k_kp*kd_t - kp_qp*kd_t - k_qp*kpd_t +4*xi*k_kp*kP_t - 2*xi*kp_qp*kP_t - 2*xi*k_qp*kpP_t) - D_minus*(k_kp*qpd_t - kp_qp*kd_t - k_qp*kpd_t + 2*xi*k_kp*qpP_t - 2*xi*kp_qp*kpP_t - 2*xi*k_qp*kpP_t))

        return A_UU, B_UU, C_UU

    def calculate_cross_section(self, phi: 'np.array')->'iterator':
        """
           Generator for the DVCS cross section as a function of phi
        """

        self.update_elastic_form_factors(-self.t[0])
        for p in phi:
            conversion = math.pow(.197326, 2)*1e7*(2*math.pi)
            A, B, C = self.calculate_kinematics(0, p)
            A_term = A*conversion*(self.GAMMA/(-self.t[0]*self.Q2[0]))*(self.F1*self.ReH[0] + self.tau*self.F2*self.ReE[0])
            B_term = B*conversion*(self.GAMMA/(-self.t[0]*self.Q2[0]))*(self.F1 + self.F2)*(self.ReH[0] + self.ReE[0])
            C_term = C*conversion*(self.GAMMA/(-self.t[0]*self.Q2[0]))*(self.F1 + self.F2)*self.ReHt[0]

#            print('A: {0}\nB: {1}\nC: {2}\n'.format(A_term, B_term, C_term))

            self.cross_section = (A_term + B_term + C_term)
            yield self.cross_section

    def contract(self, a: 'np.array', b: 'numpy array', **kawgs)->'float':

        if 'type' in kawgs:
            if kawgs['type'] is 'transverse':
                return(a[1]*b[1] + a[2]*b[2])
            else:
                pass
        else:
         return(np.dot(a, np.dot(self.metric, b)))

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


if __name__ == "__main__":
    
    femto = FemtoGen()
    femto.read_data_file('data\cff.csv')
    
    phi = math.radians(360)*np.array([np.random.random() for i in range(1000)])
    cs = -np.fromiter(femto.calculate_cross_section(phi), dtype=float, count = phi.size)

    plt.scatter(phi, cs, marker='.')
    plt.grid(True)
    plt.xlabel(r'$\phi$(radians)')
    plt.ylabel(r'$\frac{{d^{5}\sigma^{I}_{unpolar}}}{dx_{bj} dQ^{2} dt d\phi d\phi_{s}}$', fontsize='x-large')
    plt.ylim(-0.01, 0.025)
    plt.xlim(0, 6.29)

    plt.show()
