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

from matplotlib.mlab import find
import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy import interpolate


def boundary_conditions(x, t, z=None, nbsym=2):
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
    """
    indmax = argrelmax(x)[0]
    indmin = argrelmin(x)[0]
    lx = x.shape[0] - 1
    if indmin.shape[0] + indmax.shape[0] < 3:
        raise ValueError("Not enough extrema.")

    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = indmax[1:np.min([indmax.shape[0], nbsym + 1])][::-1]
            lmin = indmin[:np.min([indmin.shape[0], nbsym])][::-1]
            lsym = indmax[0]
        else:
            lmax = indmax[1:np.min([indmax.shape[0], nbsym])][::-1]
            lmin = indmin[:np.min([indmin.shape[0], nbsym - 1])][::-1]
            lmin = np.hstack((lmin, [1]))
            lsym = 1
    else:
        if x[0] < x[indmax[0]]:
            lmax = indmax[:np.min([indmax.shape[0], nbsym])][::-1]
            lmin = indmin[1:np.min([indmin.shape[0], nbsym + 1])][::-1]
            lsym = indmin[0]
        else:
            lmax = indmax[:np.min([indmin.shape[0], nbsym - 1])][::-1]
            lmax = np.hstack((lmax, [1]))
            lmin = indmin[:np.min([indmax.shape[0], nbsym])][::-1]
            lsym = 1

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = indmax[(max([indmax.shape[0] - nbsym + 1, 1]) - 1):][::-1]
            rmin = indmin[(max([indmin.shape[0] - nbsym, 1]) - 1):-1][::-1]
            rsym = indmin[-1]
        else:
            rmax = indmax[max(indmax.shape[0] - nbsym + 1, 0):indmax.shape[0]][::-1]
            rmax = np.hstack(([lx], rmax))
            rmin = indmin[max(indmin.shape[0] - nbsym, 0):][::-1]
            rsym = lx
    else:
        if x[-1] > x[indmin[-1]]:
            rmax = indmax[max(indmax.shape[0] - nbsym - 1, 0):-1][::-1]
            rmin = indmin[max(indmin.shape[0] - nbsym, 0):][::-1]
            rsym = indmax[-1]
        else:
            rmax = indmax[max(indmax.shape[0] - nbsym, 0):][::-1]
            rmin = indmin[max(indmin.shape[0] - nbsym + 1, 0):][::-1]
            rmin = np.hstack(([lx], rmin))
            rsym = lx

    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    # In case symmetrized parts do not extend enough
    if (tlmin[0] > t[0]) or (tlmax[0] > t[1]):
        if lsym == indmax[0]:
            lmax = indmax[:np.min((indmax.shape[0], nbsym))][::-1]
        else:
            lmin = indmin[:np.min((indmin.shape[0], nbsym))][::-1]
        if lsym == 1:
            raise Exception("Bug")
        lsym = 1
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if (trmin[-1] < t[lx]) or (trmax[-1] < t[lx]):
        if rsym == indmax.shape[0]:
            rmax = indmax[np.max([indmax.shape[0] - nbsym + 1,
                                 1]):indmax.shape[0]][::-1]
        else:
            rmin = indmin[np.max([indmax.shape[0] - nbsym + 1,
                                 1]):indmin.shape[0]][::-1]

        if rsym == lx:
            raise Exception("bug")
        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    if z is None:
        z = x
    zlmax = z[lmax]
    zlmin = z[lmin]
    zrmax = z[rmax]
    zrmin = z[rmin]

    tmin = map(np.array, [tlmin, t[indmin], trmin])
    tmax = map(np.array, [tlmax, t[indmax], trmax])
    zmin = map(np.array, [zlmin, z[indmin], zrmin])
    zmax = map(np.array, [zlmax, z[indmax], zrmax])

    tmin, tmax, zmin, zmax = map(np.hstack, [tmin, tmax, zmin, zmax])
    return tmin, tmax, zmin, zmax


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


def extr(x):
    """Extract the indices of the extrema and zero crossings.

    :param x: input signal
    :type x: array-like
    :return: indices of minima, maxima and zero crossings.
    :rtype: tuple
    """
    m = x.shape[0]

    x1 = x[:m - 1]
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
