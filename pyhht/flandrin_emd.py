# -*- coding: utf-8 -*-
#References
#
#
# [1] N. E. Huang et al., "The empirical mode decomposition and the
# Hilbert spectrum for non-linear and non stationary time series analysis",
# Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
#
# [2] G. Rilling, P. Flandrin and P. Gonçalves
# "On Empirical Mode Decomposition and its algorithms",
# IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing
# NSIP-03, Grado (I), June 2003
#
# [3] G. Rilling, P. Flandrin, P. Gonçalves and J. M. Lilly.,
# "Bivariate Empirical Mode Decomposition",
# Signal Processing Letters (submitted)
#
# [4] N. E. Huang et al., "A confidence limit for the Empirical Mode
# Decomposition and Hilbert spectral analysis",
# Proc. Royal Soc. London A, Vol. 459, pp. 2317-2345, 2003
#
# [5] R. Deering and J. F. Kaiser, "The use of a masking signal to improve 
# empirical mode decomposition", ICASSP 2005

import warnings
import numpy as np
from math import pi
from matplotlib.mlab import find
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev





if __name__ == "__main__":
    Fs = 10000.0
    ts = np.linspace(0, 1, Fs)
    f1, f2 = 5, 10
    y1 = np.sin(2*np.pi*f1*ts)
    y2 = np.sin(2*np.pi*f2*ts)
    y = y1 + y2
    x = y
    x += np.linspace(0, 1, x.shape[0])

    plt.plot(x)
    plt.show()
    emd = EMD(x)
