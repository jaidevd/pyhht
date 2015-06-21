import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy import angle, unwrap

from math import pi

x = np.linspace(-5 * pi, 5 * pi, 10000)
s1 = np.sin(x)
s2 = np.sin(x) - 0.5
s3 = np.sin(x) + 2

hs1 = hilbert(s1)
hs2 = hilbert(s2)
hs3 = hilbert(s3)

omega_s1 = unwrap(angle(hs1))  # unwrapped instantaneous phase
omega_s2 = unwrap(angle(hs2))
omega_s3 = unwrap(angle(hs3))

f_inst_s1 = np.diff(omega_s1)  # instantaneous frequency
f_inst_s2 = np.diff(omega_s2)
f_inst_s3 = np.diff(omega_s3)

plt.plot(x[1:], f_inst_s1, "b")
plt.plot(x[1:], f_inst_s2, "g")
plt.plot(x[1:], f_inst_s3, "r")
plt.xlim(-5 * np.pi, 5 * np.pi)
plt.grid()
plt.show()
