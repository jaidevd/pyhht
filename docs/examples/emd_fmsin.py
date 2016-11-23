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
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs


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
plot_imfs(x, imf, t)
