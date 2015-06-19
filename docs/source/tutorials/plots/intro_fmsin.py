from tftb.generators import fmsin
import numpy as np
import matplotlib.pyplot as plt
sig, iflaw = fmsin(256, 0.1, 0.3, period=64)
plt.plot(np.real(sig))
plt.grid()
plt.xlim(0, 256)
plt.show()
