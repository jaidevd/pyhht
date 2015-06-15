#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Empirical Mode Decomposition."""

import numpy as np
from numpy import pi
import warnings
from scipy.interpolate import splrep, splev
from utils import extr, boundary_conditions


def emd(data, extrapolation='mirror', nimfs=12, shifting_distance=0.2):
    """
    Perform a Empirical Mode Decomposition on a data set.

    This function will return an array of all the Imperical Mode Functions as
    defined in [1]_, which can be used for further Hilbert Spectral Analysis.

    The EMD uses a spline interpolation function to approcimate the upper and
    lower envelopes of the signal, this routine implements a extrapolation
    routine as described in [2]_ as well as the standard spline routine.
    The extrapolation method removes the artifacts introduced by the spline fit
    at the ends of the data set, by making the dataset a continuious circle.

    :param data: Signal data
    :param extrapolation: Extrapolation method for edge effects.
    :param nimfs:  Number fo IMFs to be found
    :param shifting_distance: Sets the minimum variance between iterations.
    :type data: array_like
    :type extrapolation: str
    :type nimfs: int
    :type shifting_distance: float
    :return: An array of shape (len(data), ) where N is the number of IMFs
    :rtype: array_like

    :References:

    .. [1] Huang H. et al. 1998 'The empirical mode decomposition and the \
            Hilbert spectrum for nonlinear and non-stationary time series \
            analysis.' \
            Procedings of the Royal Society 454, 903-995

    .. [2] Zhao J., Huang D. 2001 'Mirror extending and circular spline \
            function for empirical mode decomposition method'. \
            Journal of Zhejiang University (Science) V.2, No.3, 247-252

    .. [3] Rato R.T., Ortigueira M.D., Batista A.G 2008 'On the HHT, its \
            problems, and some solutions.' \
            Mechanical Systems and Signal Processing 22 1374-1394
    """

    # Set up signals array and IMFs array based on type of extrapolation
    # No extrapolation and 'extend' use signals array which is len(data)
    # Mirror extrapolation (Zhao 2001) uses a signal array len(2*data)
    if not(extrapolation):
        base = len(data)
        signals = np.zeros([base, 2])
        nimfs = range(nimfs)
        IMFs = np.zeros([base, len(nimfs)])
        ncomp = 0
        residual = data
        signals[:, 0] = data
        # DON'T do spline fitting with periodic bounds
        inter_per = 0

    elif extrapolation == 'mirror':
        # Set up base
        base = len(data)
        nimfs = range(nimfs)  # Max number of IMFs
        IMFs = np.zeros([base, len(nimfs)])
        ncomp = 0
        residual = data
        # Signals is 2*base
        signals = np.zeros([base * 2, 2])
        # Mirror Dataset
        signals[0:base / 2, 0] = data[::-1][base / 2:]
        signals[base / 2:base + base / 2, 0] = data
        signals[base + base / 2:base * 2, 0] = data[::-1][0:base / 2]
        # Redfine base as len(signals) for IMFs
        base = len(signals)
        data_length = len(data)  # Data length is used in recovering input data
        # DO spline fitting with periodic bounds
        inter_per = 1

    else:
        raise Exception(
                "Please Specifiy extrapolation keyword as None or 'mirror'")

    for j in nimfs:
#       Extract at most nimfs IMFs no more IMFs to be found when Finish is True
        k = 0
        sd = 1.
        finish = False

        while sd > shifting_distance and not(finish):
            min_env = np.zeros(base)
            max_env = min_env.copy()

            min_env = np.logical_and(
                                np.r_[True, signals[1:, 0] > signals[:-1, 0]],
                                np.r_[signals[:-1, 0] > signals[1:, 0], True])
            max_env = np.logical_and(
                                np.r_[True, signals[1:, 0] < signals[:-1, 0]],
                                np.r_[signals[:-1, 0] < signals[1:, 0], True])
            max_env[0] = max_env[-1] = False
            min_env = min_env.nonzero()[0]
            max_env = max_env.nonzero()[0]

            # Cubic Spline by default
            order_max = 3
            order_min = 3

            if len(min_env) < 2 or len(max_env) < 2:
                # If this IMF has become a straight line
                finish = True
            else:
                if len(min_env) < 4:
                    # Do linear interpolation if not enough points
                    order_min = 1

                if len(max_env) < 4:
                    # Do linear interpolation if not enough points
                    order_max = 1

# Mirror Method requires per flag = 1 No extrapolation requires per flag = 0
# This is set in intial setup at top of function.
                t = splrep(min_env, signals[min_env, 0], k=order_min,
                           per=inter_per)
                top = splev(np.arange(len(signals[:, 0])), t)

                b = splrep(max_env, signals[max_env, 0], k=order_max,
                           per=inter_per)
                bot = splev(np.arange(len(signals[:, 0])), b)

            # Calculate the Mean and remove from the data set.
            mean = (top + bot) / 2
            signals[:, 1] = signals[:, 0] - mean

            # Calculate the shifting distance which is a measure of
            # simulartity to previous IMF
            if k > 0:
                sd = (np.sum((np.abs(signals[:, 0] - signals[:, 1])**2)) /
                      (np.sum(signals[:, 0]**2)))

            # Set new iteration as previous and loop
            signals = signals[:, ::-1]
            k += 1

        if finish:
            # If IMF is a straight line we are done here.
            IMFs[:, j] = residual
            ncomp += 1
            break

        if not(extrapolation):
            IMFs[:, j] = signals[:, 0]
            # For j==0 residual is initially data
            residual = residual - IMFs[:, j]
            signals[:, 0] = residual
            ncomp += 1

        elif extrapolation == 'mirror':
            IMFs[:, j] = signals[(data_length / 2): (data_length + data_length / 2), :]
            # For j==0 residual is initially data
            residual = residual - IMFs[:, j]

            # Mirror case requires IMF subtraction from data range then
            # re-mirroring for each IMF
            signals[0: data_length / 2, 0] = residual[::-1][data_length / 2:]
            signals[(data_length / 2): data_length + data_length / 2, 0] = residual
            signals[data_length + data_length / 2:, 0] = residual[::-1][0:data_length / 2]
            ncomp += 1

        else:
            raise Exception(
                "Please Specifiy extrapolation keyword as None or 'mirror'")

    return IMFs[:, 0:ncomp]


class EMD(object):
    """The EMD class."""

    def __init__(self, x, t=None, threshold_1=0.05, threshold_2=0.5, alpha=0.05,
                 is_mode_complex=None, ndirs=4, fixe=0, maxiter=2000,
                 fixe_h=0, n_imfs=0, nbsym=2):
        r"""Empirical mode decomposition.

        :param x: A vector on which to perform empirical mode decomposition.
        :param t: Sampling time instants.
        :param threshold_1: Threshold for the stopping criterion, corresponding to
            :math:`\theta_{1}` in [1] (Default: 0.05)
        :param threshold_2: Threshold for the stopping criterion, corresponding to
            :math:`\theta_{2}` in [1] (Default: 0.5)
        :param alpha: Tolerance for the stopping criterion, corresponding to
            :math:`\alpha` in [1] (Default: 0.05)
        :param is_mode_complex: Whether the input signal is complex.
        :param ndirs: Number of directions in which envelopes are computed.
            (Default: 4)
        :param fixe: Number of sifting iterations to perform for each mode. The
            default value is ``None``, in which case the default stopping criterion
            is used. If not ``None``, each mode will be a result of exactly
            ``fixe`` sifting iterations.
        :param maxiter: Number of maximum sifting iterations for the
            computation of each mode. (Default: 2000)
        :param fixe_h:
        :param n_imfs: Number if IMFs to extract.
        :param nbsym: Number of points to mirror when calculating envelopes.
        :type x: array-like
        :type t: array-like
        :type threshold_1: float
        :type threshold_2: float
        :type alpha: float
        :type is_mode_complex: bool
        :type ndirs: int
        :type fixe: int
        :type maxiter: int
        :type fixe_h: int
        :type n_imfs: int
        :type nbsym: int
        :return: Array of shape [n_imfs + 1, length(x)]
        :rtype: numpy.ndarray
        :Example:

        >>> from pyhht.visualization import plot_imfs
        >>> t = linspace(0, 1, 1000)
        >>> modes = sin(2 * pi * 5 * t) + sin(2 * pi * 10 * t)
        >>> x = modes + t
        >>> decomposer = EMD(x)
        >>> imfs = decomposer.decompose()
        >>> plot_imfs(x, t, imfs)

        .. plot:: ../../docs/examples/simple_emd.py
        """

        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.alpha = alpha
        self.maxiter = maxiter
        self.fixe_h = fixe_h
        self.ndirs = ndirs
        self.complex_version = 2
        self.nbit = 0
        self.Nbit = 0
        self.n_imfs = n_imfs
        self.k = 1
        # self.mask = mask
        self.nbsym = nbsym
        self.nbit = 0
        self.NbIt = 0

        if x.ndim > 1:
            if 1 not in x.shape:
                raise TypeError("x must have only one row or one column.")
        if x.shape[0] > 1:
            x = x.ravel()
        if not np.all(np.isfinite(x)):
            raise TypeError("All elements of x must be finite.")
        self.x = x
        self.ner = self.nzr = len(self.x)
        self.residue = self.x.copy()

        if t is None:
            self.t = np.arange(np.max(x.shape))
        else:
            if t.shape != self.x.shape:
                raise TypeError("t must have the same dimensions as x.")
            if t.ndim > 1:
                if 1 not in t.shape:
                    raise TypeError("t must have only one column or one row.")
            if not np.all(np.isreal(t)):
                raise TypeError("t must be a real vector.")
            if t.shape[0] > 1:
                t = t.ravel()
            self.t = t

        self.sdt = self.threshold_1 * np.ones((len(self.x),))
        self.sd2t = self.threshold_2 * np.ones((len(self.x),))

        if fixe:
            self.maxiter = fixe
            if self.fixe_h:
                raise TypeError("Cannot use both fixe and fixe_h modes")
        self.fixe = fixe

        # FIXME: `is_mode_complex` should be a boolean and self.complex_version
        # should be a string for better readability. Also, the boolean should
        # be redundant in the signature of __init__
        if is_mode_complex is None:
            is_mode_complex = not(np.all(np.isreal(self.x) * self.complex_version))
        self.is_mode_complex = is_mode_complex

        self.imf = []
        self.nbits = []

        # FIXME: Masking disabled because it seems to be recursive.
#        if np.any(mask):
#            if mask.shape != x.shape:
#                raise TypeError("Masking signal must have the same dimensions" +
#                                "as the input signal x.")
#            if mask.shape[0]>1:
#                mask = mask.ravel()
#            imf1 = emd(x+mask, opts)

    def io(self):
        r"""Compute the index of orthoginality, as defined by:

            .. math:: \sum_{i, j=1, i\neq j}^{N} \frac{\|C_{i}\overline{C_{j}}\|}{\|x\|^2}

        Where :math:`C_{i}` is the :math:`i` th IMF.

        :return: Index of orthogonality.
        :rtype: float
        :Example:

        >>> t = linspace(0, 1, 1000)
        >>> modes = sin(2 * pi * 5 * t) + sin(2 * pi * 10 * t)
        >>> x = modes + t
        >>> decomposer = EMD(x)
        >>> decomposer.decompose()
        >>> print decomposer.io()
        0.0516420404972
        """

        n = len(self.imf)
        s = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    s += np.abs(np.sum(self.imf[i] * np.conj(self.imf[j])) / np.sum(self.x**2))
        return 0.5 * s

    def stop_EMD(self):
        """Check if there are enough extrema (3) to continue sifting."""
        if self.is_mode_complex:
            ner = []
            for k in range(self.ndirs):
                phi = k * pi / self.ndirs
                indmin, indmax, _ = extr(np.real(np.exp(1j * phi) * self.residue))
                ner.append(len(indmin) + len(indmax))
            stop = np.any(ner < 3)
        else:
            indmin, indmax, _ = extr(self.residue)
            ner = len(indmin) + len(indmax)
            stop = ner < 3
        return stop

    def mean_and_amplitude(self, m):
        """ Computes the mean of the envelopes and the mode amplitudes."""
        # FIXME: The spline interpolation may not be identical with the MATLAB
        # implementation. Needs further investigation.
        if self.is_mode_complex:
            if self.is_mode_complex == 1:
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs, len(self.t)))
                envmax = np.zeros((self.ndirs, len(self.t)))
                for k in range(self.ndirs):
                    phi = k * pi / self.ndirs
                    y = np.real(np.exp(-1j * phi) * m)
                    indmin, indmax, indzer = extr(y)
                    nem.append(len(indmin) + len(indmax))
                    nzm.append(len(indzer))
                    tmin, tmax, zmin, zmax = boundary_conditions(y, self.t, m,
                                                                 self.nbsym)

                    f = splrep(tmin, zmin)
                    spl = splev(self.t, f)
                    envmin[k, :] = spl

                    f = splrep(tmax, zmax)
                    spl = splev(self.t, f)
                    envmax[k, :] = spl

                envmoy = np.mean((envmin + envmax) / 2, axis=0)
                amp = np.mean(abs(envmax - envmin), axis=0) / 2

            elif self.is_mode_complex == 2:
                nem = []
                nzm = []
                envmin = np.zeros((self.ndirs, len(self.t)))
                envmax = np.zeros((self.ndirs, len(self.t)))
                for k in range(self.ndirs):
                    phi = k * pi / self.ndirs
                    y = np.real(np.exp(-1j * phi) * m)
                    indmin, indmax, indzer = extr(y)
                    nem.append(len(indmin) + len(indmax))
                    nzm.append(len(indzer))
                    tmin, tmax, zmin, zmax = boundary_conditions(y, self.t, m,
                                                                 self.nbsym)
                    f = splrep(tmin, zmin)
                    spl = splev(self.t, f)
                    envmin[k, ] = np.exp(1j * phi) * spl

                    f = splrep(tmax, zmax)
                    spl = splev(self.t, f)
                    envmax[k, ] = np.exp(1j * phi) * spl

                envmoy = np.mean((envmin + envmax), axis=0)
                amp = np.mean(abs(envmax - envmin), axis=0) / 2

        else:
            indmin, indmax, indzer = extr(m)
            nem = len(indmin) + len(indmax)
            nzm = len(indzer)
            tmin, tmax, mmin, mmax = boundary_conditions(m, self.t, m, self.nbsym)

            f = splrep(tmin, mmin)
            envmin = splev(self.t, f)

            f = splrep(tmax, mmax)
            envmax = splev(self.t, f)

            envmoy = (envmin + envmax) / 2
            amp = np.abs(envmax - envmin) / 2.0

        return envmoy, nem, nzm, amp

    def stop_sifting(self, m):
        """Evaluate the stopping criteria for the current mode.

        :param m: The current mode
        :type m: array-like
        """
        # FIXME: This method needs a better name.
        if self.fixe:
            stop_sift, moyenne = self.mean_and_amplitude(), 0
        elif self.fixe_h:
            stop_count = 0
            try:
                moyenne, nem, nzm = self.mean_and_amplitude(m)[:3]

                if np.all(abs(nzm - nem) > 1):
                    stop = 0
                    stop_count = 0
                else:
                    stop_count += 1
                    stop = (stop_count == self.fixe_h)
            except:
                moyenne = np.zeros((len(m)))
                stop = 1
            stop_sift = stop
        else:
            envmoy, nem, nzm, amp = self.mean_and_amplitude(m)
            sx = np.abs(envmoy) / amp
            stop = not(((np.mean(sx > self.threshold_1) > self.alpha) or
                        np.any(sx > self.threshold_2)) and np.all(nem > 2))
            if not self.is_mode_complex:
                stop = stop and not(np.abs(nzm - nem) > 1)
            stop_sift = stop
            moyenne = envmoy
        return stop_sift, moyenne

    def keep_decomposing(self):
        """Check whether to continue the sifting operation."""
        return not(self.stop_EMD()) and \
            (self.k < self.n_imfs + 1 or self.n_imfs == 0)  # and \
# not(np.any(self.mask))

    def decompose(self):
        """Decompose the input signal into IMFs.

        This function does all the heavy lifting required for sifting, and
        should ideally be the only public method of this class."""
        while self.keep_decomposing():

            # current mode
            m = self.residue

            # computing mean and stopping criterion
            stop_sift, moyenne = self.stop_sifting(m)

            # in case current mode is small enough to cause spurious extrema
            if np.max(np.abs(m)) < (1e-10) * np.max(np.abs(self.x)):
                if not stop_sift:
                    warnings.warn("EMD Warning: Amplitude too small, stopping.")
                else:
                    print "Force stopping EMD: amplitude too small."
                return

            # SIFTING LOOP:
            while not(stop_sift) and (self.nbit < self.maxiter):

                if (not(self.is_mode_complex) and (self.nbit > self.maxiter / 5) and
                        self.nbit % np.floor(self.maxiter / 10) == 0 and
                        not(self.fixe) and self.nbit > 100):
                    print "Mode " + str(self.k) + ", Iteration " + str(self.nbit)
                    im, iM, _ = extr(m)
                    print str(np.sum(m[im] > 0)) + " minima > 0; " + str(np.sum(m[im] < 0)) + " maxima < 0."

                # Sifting
                m = m - moyenne

                # Computing mean and stopping criterion
                if self.fixe:
                    stop_sift, moyenne = self.stop_sifting_fixe()
                elif self.fixe_h:
                    stop_sift, moyenne, stop_count = self.stop_sifting_fixe_h()
                else:
                    stop_sift, moyenne = self.stop_sifting(m)

                self.nbit += 1
                self.NbIt += 1

                if (self.nbit == (self.maxiter - 1)) and not(self.fixe) and (self.nbit > 100):
                    warnings.warn("Emd:warning, Forced stop of sifting - " +
                                  "too many iterations")

            self.imf.append(m)

            self.nbits.append(self.nbit)
            self.k += 1

            self.residue = self.residue - m
            self.ort = self.io()

        if np.any(self.residue):
            self.imf.append(self.residue)
        return np.array(self.imf)
