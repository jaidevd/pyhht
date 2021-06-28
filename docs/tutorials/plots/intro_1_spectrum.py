import numpy as np
import matplotlib.pyplot as plt
from tftb.generators import fmconst
n_points = 128
mode1, iflaw1 = fmconst(n_points, fnorm=0.1)
mode2, iflaw2 = fmconst(n_points, fnorm=0.3)
signal = np.r_[mode1, mode2]

X = np.fft.fftshift(np.fft.fft(signal))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.linspace(-0.5, 0.5, 256), np.abs(X) ** 2)
ax.set_xlim(-0.5, 0.5)
ax.grid(True)
ax.set_yticklabels([])
plt.xlabel("Normalized Frequency")
ax.set_title("Energy Spectrum")
plt.show()
