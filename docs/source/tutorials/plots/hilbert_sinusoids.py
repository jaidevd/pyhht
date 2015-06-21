import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

x = np.linspace(0, 2 * np.pi, 1000)
s1 = np.sin(x)
s2 = np.sin(x) - 1
s3 = np.sin(x) + 2

hs1 = hilbert(s1)
hs2 = hilbert(s2)
hs3 = hilbert(s3)

plt.plot(np.real(hs1), np.imag(hs1), 'b')
plt.plot(np.real(hs2), np.imag(hs2), 'g')
plt.plot(np.real(hs3), np.imag(hs3), 'r')
plt.axis('equal')
plt.ylim(-1.5, 1.5)
plt.grid()
plt.show()
