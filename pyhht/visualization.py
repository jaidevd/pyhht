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


def plot_imfs(signal, imfs, time_samples=None, fignum=None, show=True):
    """
    Plot the signal, IMFs and residue.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
        The input signal.
    imfs : array-like, shape (n_imfs, n_samples)
        Matrix of IMFs as generated with the `EMD.decompose` method.
    time_samples : array-like, shape (n_samples), optional
        Time instants of the signal samples.
        (defaults to `np.arange(1, len(signal))`)
    fignum : int, optional
        Matplotlib figure number (by default a new figure is created)
    show : bool, optional
        Whether to display the plot. Defaults to True, set to False if further
        plotting needs to be done.

    Returns
    -------
    `matplotlib.figure.Figure`
        The figure (new or existing) in which the decomposition is plotted.

    Examples
    --------
    >>> from pyhht.visualization import plot_imfs
    >>> import numpy as np
    >>> from pyhht import EMD
    >>> t = np.linspace(0, 1, 1000)
    >>> modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    >>> x = modes + t
    >>> decomposer = EMD(x)
    >>> imfs = decomposer.decompose()
    >>> plot_imfs(x, imfs, t) #doctest: +SKIP

    .. plot:: ../../docs/examples/simple_emd.py

    """
    is_bivariate = np.any(np.iscomplex(signal))
    if time_samples is None:
        time_samples = np.arange(signal.shape[0])

    n_imfs = imfs.shape[0]

    fig = plt.figure(num=fignum)
    axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))

    # Plot original signal
    ax = plt.subplot(n_imfs + 1, 1, 1)
    if is_bivariate:
        ax.plot(time_samples, np.real(signal), 'b')
        ax.plot(time_samples, np.imag(signal), 'k--')
    else:
        ax.plot(time_samples, signal)
    ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal')
    ax.set_title('Empirical Mode Decomposition')

    # Plot the IMFs
    for i in range(n_imfs - 1):
        ax = plt.subplot(n_imfs + 1, 1, i + 2)
        if is_bivariate:
            ax.plot(time_samples, np.real(imfs[i]), 'b')
            ax.plot(time_samples, np.imag(imfs[i]), 'k--')
        else:
            ax.plot(time_samples, imfs[i])
        ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('imf' + str(i + 1))

    # Plot the residue
    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    if is_bivariate:
        ax.plot(time_samples, np.real(imfs[-1]), 'r')
        ax.plot(time_samples, np.imag(imfs[-1]), 'r--')
    else:
        ax.plot(time_samples, imfs[-1])
    ax.axis('tight')
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res.')

    if show:  # pragma: no cover
        plt.show()
    return fig
