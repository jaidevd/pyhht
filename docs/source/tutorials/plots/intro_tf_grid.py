from tftb.processing import ShortTimeFourierTransform
from tftb.generators import fmconst
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming

x = np.r_[fmconst(128, 0.1)[0], fmconst(128, 0.3)[0]]

n_fbins = 2 ** np.arange(3, 9)

h = np.floor(n_fbins / 4.0)
h += 1 - np.remainder(h, 2)


def make_window(wlength):
    w = hamming(wlength)
    return w / np.linalg.norm(w)

fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(12, 12),
                         subplot_kw={'xticks': [], 'yticks': []})

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        window = make_window(h[j])
        stft = ShortTimeFourierTransform(x, fwindow=window, n_fbins=n_fbins[i])
        stft.run()
        stft.plot(ax=axes[i, j], show=False, default_annotation=False)
        if j == 0:
            axes[i, j].set_ylabel(str(n_fbins[i]))
        if i == 5:
            axes[i, j].set_xlabel(str(h[j]))

fig.text(0.5, 0.04, 'Window lengths', ha='center')
fig.text(0.04, 0.5, 'Frequency bins', va='center', rotation='vertical')
plt.show()
