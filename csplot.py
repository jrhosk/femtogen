import numpy as np
import math
import matplotlib.pyplot as plt

from femtogen import FemtoGen

if __name__ == "__main__":
    
    femto = FemtoGen()
    kinematics = femto.read_data_file('data\cff.csv')
    femto.set_kinematics(array=kinematics[0])

    phi = math.radians(360)*np.array([np.random.random() for i in range(1000)])
    cs = np.fromiter(femto.generate_cross_section(phi,
                                                  error=femto.error_generator(stdev=0.005),
                                                  type='full'), dtype=float, count=phi.size)

    plt.scatter(phi, cs, marker='.', color='xkcd:barney purple')
    plt.grid(True)
    plt.xlabel(r'$\phi$(radians)')
    plt.ylabel(r'$\frac{{d^{5}\sigma^{I}_{unpolar}}}{dx_{bj} dQ^{2} dt d\phi d\phi_{s}}$', fontsize='x-large')
    plt.ylim(-0.05, 0.19)
    plt.xlim(0, 6.29)

    plt.show()