from tftb.generators import fmconst, amgauss
import matplotlib.pyplot as plt
from numpy import real
x = amgauss(128) * fmconst(128)[0]
plt.plot(real(x))
plt.grid()
plt.xlim(0, 128)
plt.title("Gaussian amplitude modulation")
plt.show()
