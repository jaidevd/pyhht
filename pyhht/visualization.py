#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""Visualization functions for PyHHT."""


import matplotlib.pyplot as plt
import numpy as np


def plot_imfs(signal, time_samples, imfs, fignum=None):
    """Visualize decomposed signals.

    :param signal: Analyzed signal
    :param time_samples: time instants
    :param imfs: intrinsic mode functions of the signal
    :param fignum: (optional) number of the figure to display
    :type signal: array-like
    :type time_samples: array-like
    :type imfs: array-like of shape (n_imfs, length_of_signal)
    :type fignum: int
    :return: None
    :Example:

    >>> plot_imfs(signal)

    .. plot:: ../../docs/examples/emd_fmsin.py
    """

    n_imfs = imfs.shape[0]

    plt.figure(num=fignum)
    axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))

    # Plot original signal
    ax = plt.subplot(n_imfs, 1, 1)
    ax.plot(time_samples, signal)
    ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
            labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal')
    ax.set_title('Empirical Mode Decomposition')

    # Plot the IMFs
    for i in range(n_imfs - 1):
        ax = plt.subplot(n_imfs, 1, i + 2)
        ax.plot(time_samples, imfs[i, :])
        ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('imf' + str(i + 1))

    # Plot the residue
    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    ax.plot(time_samples, imfs[-1, :], 'r')
    ax.axis('tight')
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
            labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res.')

    plt.show()
