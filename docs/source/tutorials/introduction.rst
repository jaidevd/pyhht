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

This first half of the signal is a sinusoid with a normalized frequency of 0.1
and the other half has a normalized frequency of 0.3. If we look at the energy
spectrum of this signal, sure enough, there are two peaks at the respective
frequency::

    >>> X = np.fft.fftshift(np.fft.fft(signal))
    >>> plt.plot(np.linspace(-0.5, 0.5, 256), np.abs(X) ** 2)

.. plot:: tutorials/plots/intro_1_spectrum.py

Note that the signal produced by the ``fmconst`` function produces an `Analytic
Signal <https://en.wikipedia.org/wiki/Analytic_signal>`_, which are complex
valued, and by definition do not have negative frequency components.

A note on time-frequency analysis
+++++++++++++++++++++++++++++++++

The energy spectrum is perfectly valid, but the Fourier transform is
essentially an integral over time. Thus, we lose all information that varies
with time. All we can tell from the spectrum is that the signal has two
distinct frequency components. In other words, we can comment on *what*
happens a signal, not *when* it happens. Consider a song as the signal under
consideration. If you were not interested in time, the whole point of
processing that signal would be lost. Rhythm and timing are the very heart of
good music, after all. In this case, we want
to know when the drums kicked in, as well as what notes were being played on
the guitar. If we perform only frequency analysis, all time information would
be lost and the only information we would have would be about what frequencies
were played in the song, and what their respective amplitudes were, averaged
over the duration of the entire song. So even if the drums stop playing after
the second stanza, the frequency spectrum would show them playing throughout
the song. Conversely, if we were only interested in the time information, we
would be hardly better off than simply listening to the song.

The solution to this is `time-frequency
analysis <https://en.wikipedia.org/wiki/Time%E2%80%93frequency_analysis>`_, which
is a field that deals with signal processing in both time and frequency domain.
It consists of a collection of methods that allow us to make tradeoffs between
time and frequency processing of a signal, depending on what makes more sense
for a particular application. HHT too is a tool for time-frequency analysis,
as we shall see.

Time-Frequency representations of the signal
++++++++++++++++++++++++++++++++++++++++++++

A popular choice to represent both time and frequency characteristics is the
`short-time Fourier transform (STFT)
<https://en.wikipedia.org/wiki/Short-time_Fourier_transform>`_, which, simply
put, transforms contiguous chunks of the input and aggregates the result in a 2
dimensional form, where one axis represents frequency and the other represents
time. PyTFTB has an STFT implementation which we can use as follows::

    >>> from tftb.processing import ShortTimeFourierTransform
    >>> stft = ShortTimeFourierTransform(signal)
    >>> stft.run()
    >>> stft.plot()

.. plot:: tutorials/plots/intro_2.py
