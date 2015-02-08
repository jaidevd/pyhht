#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Utility functions used to inspect EMD functionality.
"""

import matplotlib.pyplot as plt
from matplotlib.mlab import find
import numpy as np
from scipy.signal import argrelmax, argrelmin


def get_envelops(x, t=None):
    """ Find the upper and lower envelopes of the array `x`.
    """
    if t is None:
        t = np.arange(x.shape[0])
    maxima = argrelmax(x)[0]
    minima = argrelmin(x)[0]

    # consider the start and end to be extrema

    ext_maxima = np.zeros((maxima.shape[0] + 2,), dtype=int)
    ext_maxima[1:-1] = maxima
    ext_maxima[0] = 0
    ext_maxima[-1] = t.shape[0] - 1

    ext_minima = np.zeros((minima.shape[0] + 2,), dtype=int)
    ext_minima[1:-1] = minima
    ext_minima[0] = 0
    ext_minima[-1] = t.shape[0] - 1

    tck = interpolate.splrep(t[ext_maxima], x[ext_maxima])
    upper = interpolate.splev(t, tck)
    tck = interpolate.splrep(t[ext_minima], x[ext_minima])
    lower = interpolate.splev(t, tck)
    return upper, lower


def plot_imfs(imfs, shape=None):
    if shape is None:
        shape = imfs.shape[1], 1
    for i in range(imfs.shape[1]):
        plt.subplot(shape[0], shape[1], i+1)
        plt.plot(imfs[:,i])
    plt.show()


def extr(x):
    """ Extracts the indices of the extrema and zero crossings. """
    # FIXME: This doesn't have to be a method here.
    m = x.shape[0]

    x1 = x[:m-1]
    x2 = x[1:m]
    indzer = find(x1 * x2 < 0)
    if np.any(x == 0):
        iz = find(x == 0)
        indz = []
        if np.any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff([0, zer, 0])
            debz = find(dz == 1)
            finz = find(dz == -1) - 1
            indz = np.round((debz + finz) / 2)
        else:
            indz = iz
        indzer = np.sort(np.hstack([indzer, indz]))

    indmax = argrelmax(x)[0]
    indmin = argrelmin(x)[0]

    return indmin, indmax, indzer
