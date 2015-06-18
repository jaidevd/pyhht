import numpy as np
import matplotlib.pyplot as plt
from tftb.generators import fmconst
n_points = 128
mode1, iflaw1 = fmconst(n_points, fnorm=0.1)
mode2, iflaw2 = fmconst(n_points, fnorm=0.3)
signal = np.r_[mode1, mode2]
plt.plot(np.real(signal)), plt.xlim(xmax=256)
plt.grid(), plt.show()
