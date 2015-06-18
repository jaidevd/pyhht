Introduction to the Hilbert Huang Transform with PyHHT
======================================================

The `Hilbert Huang transform
<https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform>`_
(HHT) is a time series analysis technique that is
designed to handle nonlinear and nonstationary time series data. PyHHT is a
Python module based on NumPy and SciPy which implements the HHT. This tutorial
introduces HHT, the common vocabulary associated with it and the usage of the
PyHHT module itself to analyze time series data.


Motivation for the Hilbert Huang Transform
------------------------------------------

To begin with, let us construct a nonstationary signal, and try to glean its
time and frequency characteristics. Consider the signal obtained as follows::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from tftb.generators import fmconst
    >>> n_points = 128
    >>> mode1, iflaw1 = fmconst(n_points, fnorm=0.1)
    >>> mode2, iflaw2 = fmconst(n_points, fnorm=0.3)
    >>> signal = np.r_[mode1, mode2]
    >>> plt.plot(np.real(signal)), plt.grid(), plt.show()

.. plot:: tutorials/plots/intro_1.py
.. plot:: tutorials/plots/intro_2.py
