#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
from tftb.generators import fmsin, fmconst, amgauss
from scipy.signal import kaiser
from tftb.processing.reassigned import spectrogram
from pyhht.emd import EMD
import matplotlib.pyplot as plt


N = 2001
T = np.arange(1, N + 1, step=4)
t = np.arange(1, N + 1)

p = N / 2

fmin1 = 1.0 / 64
fmax1 = 1.5 * 1.0 / 8
x1 = fmsin(N, fmin1, fmax1, p, N / 2, fmax1)[0]

fmin2 = 1.0 / 32
fmax2 = 1.5 * 1.0 / 4
x2 = fmsin(N, fmin2, fmax2, p, N / 2, fmax2)[0]

f0 = 1.5 * 1.0 / 16

x3 = amgauss(N, N / 2, N / 8) * fmconst(N, f0)[0]

a1 = 1
a2 = 1
a3 = 1

x = np.real(a1 * x1 + a2 * x2 + a3 * x3)
x = x / np.max(np.abs(x))

decomposer = EMD(x)
imf = decomposer.decompose()

n_freq_bins = 256
short_window_length = 127
beta = 3 * np.pi
window = kaiser(short_window_length, beta=beta)

_, re_spec_sig, _ = spectrogram(x, t, n_freq_bins, window)
_, re_spec_imf1, _ = spectrogram(imf[0, :], t, n_freq_bins, window)
_, re_spec_imf2, _ = spectrogram(imf[1, :], t, n_freq_bins, window)
_, re_spec_imf3, _ = spectrogram(imf[2, :], t, n_freq_bins, window)

fig = plt.figure()
for i, rspec in enumerate([re_spec_sig, re_spec_imf1, re_spec_imf2,
                           re_spec_imf3]):
    rspec = np.abs(rspec)[:128, :]
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(np.flipud(rspec), extent=[0, 1, 0, 1])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
            labelbottom=False)
    ax.set_xlabel('time')
    ax.set_ylabel('frequency')
    if i == 0:
        ax.set_title('signal')
    else:
        ax.set_title('mode #{}'.format(i))
plt.show()
