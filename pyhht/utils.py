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

import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy import interpolate, angle


def inst_freq(x, t=None, L=1):
    """
    Compute the instantaneous frequency of an analytic signal at specific
    time instants using the trapezoidal integration rule.

    :param x: The input analytic signal
    :param t: The time instants at which to calculate the instantaneous frequencies.
    :param L: Non default values are currently not supported.
        If L is 1, the normalized instantaneous frequency is computed. If L > 1,
        the maximum likelihood estimate of the instantaneous frequency of the
        deterministic part of the signal.
    :type x: numpy.ndarray
    :type t: numpy.ndarray
    :type L: int
    :return: instantaneous frequencies of the input signal.
    :rtype: numpy.ndarray
    :Example:
    >>> from tftb.generators import fmsin
    >>> import matplotlib.pyplot as plt
    >>> x = fmsin(70, 0.05, 0.35, 25)[0]
    >>> instf, timestamps = inst_freq(x)
    >>> plt.plot(timestamps, instf) #doctest: +SKIP

    .. plot:: docstring_plots/utils/inst_freq.py
    """
    if x.ndim != 1:
        if 1 not in x.shape:
            raise TypeError("Input should be a one dimensional array.")
        else:
            x = x.ravel()
    if t is not None:
        if t.ndim != 1:
            if 1 not in t.shape:
                raise TypeError("Time instants should be a one dimensional "
                                "array.")
            else:
                t = t.ravel()
    else:
        t = np.arange(2, len(x))

    fnorm = 0.5 * (angle(-x[t] * np.conj(x[t - 2])) + np.pi) / (2 * np.pi)
    return fnorm, t


def boundary_conditions(signal, time_samples, z=None, nbsym=2):
    """
    Extend the signal beyond it's bounds w.r.t mirror symmetry.

    :param x: Signal to be mirrored.
    :param t: Timestamps of the signal
    :param z: Signal on whose extrema the interpolation is evaluated. (By \
        default this is just ``x``)
    :param nbsym: Number of points added to each end of the signal.
    :type x: array-like
    :type t: array-like
    :type z: array-like
    :type nbsym: int
    :return: timestamps and values of extended extrema, ordered as (minima \
        timestamps, maxima timestamps, minima values, maxima values.)
    :rtype: tuple
    :Example:
    >>> from __future__ import print_function
    >>> import numpy as np
    >>> signal = np.array([-1, 1, -1, 1, -1])
    >>> print(boundary_conditions(signal, np.arange(5)))
    (array([-2,  2,  6]), array([-3, -1,  1,  3,  5,  7]), array([-1, -1, -1]), array([1, 1, 1, 1, 1, 1]))
    """
    tmax = argrelmax(signal)[0]
    maxima = signal[tmax]
    tmin = argrelmin(signal)[0]
    minima = signal[tmin]

    if tmin.shape[0] + tmax.shape[0] < 3:
        raise ValueError("Not enough extrema.")

    loffset_max = time_samples[tmax[:nbsym]] - time_samples[0]
    roffset_max = time_samples[-1] - time_samples[tmax[-nbsym:]]
    new_tmax = np.r_[time_samples[0] - loffset_max[::-1], time_samples[tmax], roffset_max[::-1] + time_samples[-1]]
    new_vmax = np.r_[maxima[:nbsym][::-1], maxima, maxima[-nbsym:][::-1]]

    loffset_min = time_samples[tmin[:nbsym]] - time_samples[0]
    roffset_min = time_samples[-1] - time_samples[tmin[-nbsym:]]

    new_tmin = np.r_[time_samples[0] - loffset_min[::-1], time_samples[tmin], roffset_min[::-1] + time_samples[-1]]
    new_vmin = np.r_[minima[:nbsym][::-1], minima, minima[-nbsym:][::-1]]
    return new_tmin, new_tmax, new_vmin, new_vmax


def get_envelops(x, t=None):
    """ Find the upper and lower envelopes of the array `x`.
    :Example:
    >>> import numpy as np
    >>> x = np.random.rand(100,)
    >>> upper, lower = get_envelops(x)
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


def extr(x):
    """Extract the indices of the extrema and zero crossings.

    :param x: input signal
    :type x: array-like
    :return: indices of minima, maxima and zero crossings.
    :rtype: tuple
    :Example:
    >>> from __future__ import print_function
    >>> import numpy as np
    >>> x = np.array([0, -2, 0, 1, 3, 0.5, 0, -1, -1])
    >>> indmin, indmax, indzer = extr(x)
    >>> print(indmin)
    [1]
    >>> print(indmax)
    [4]
    >>> print(indzer)
    [0 2 6]
    """
    m = x.shape[0]

    x1 = x[:m - 1]
    x2 = x[1:m]
    indzer = np.where(x1 * x2 < 0)[0]
    if np.any(x == 0):
        iz = np.where(x == 0)[0]
        indz = []
        if np.any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff(np.r_[0, zer, 0])
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2)
        else:
            indz = iz
        indzer = np.sort(np.hstack([indzer, indz]))

    indmax = argrelmax(x)[0]
    indmin = argrelmin(x)[0]

    return indmin, indmax, indzer
