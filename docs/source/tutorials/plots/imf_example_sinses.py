import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 1000)
s1 = np.sin(x)
s2 = np.sin(x) - 1
s3 = np.sin(x) + 2
plt.plot(x, s1, 'b', x, s2, 'g', x, s3, 'r')
plt.xlim(0, x.max())
plt.ylim(-2, 3)
plt.grid()
plt.show()
