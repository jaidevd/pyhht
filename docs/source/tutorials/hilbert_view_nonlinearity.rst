Motivation for Hilbert Spectral Analysis
========================================

The Fourier transform generalizes Fourier coefficients of a signal over time.
Since the Fourier coefficients are the measures of the signal amplitude as a
function of frequency, the time information is totally lost, as we saw in the
`last section
<http://pyhht.readthedocs.org/en/latest/tutorials/limitations_fourier.html>`_.
To address this issue there have developed further modifications of the
Fourier transform, the most popular of which is the
`short-time Fourier transform (STFT)
<https://en.wikipedia.org/wiki/Short-time_Fourier_transform>`_. The STFT
divides the input signal into windows of time and then considers the Fourier
transforms of those time windows, thereby achieving some localization of
frequency information along the time axis. While practically powerful for
most signals, this method cannot be generalized for a broad class of signals
because of its *a priori* window lengths. Particularly, the window lengths
must be long enough to capture at least one cycle of a component frequency,
but not so long as to be redundant. On the other hand, most real-life signals
are nonstationary, or have multiple frequency components. The duration of the
STFT windows should not be so long as to mix the multiple components during a
single operation of the kernel. This might lead to highly undesirable results
like the frequency analysis representing multiple components of a
nonstationary signal as harmonics of lower components.

A powerful variant of the Fourier transform is the wavelet transform. By
using finite-support basis functions, wavelets are able to approximate even
nonstationary data. These basis functions possess most of the desirable
properties required for linear decomposition (like orthogonality, completenes
, etc) and they can be drawn from a large dictionary of wavelets. This makes
the wavelet transform a versatile tool for analysis of nonstationary data.
But the wavelet transform is still a linear decomposition and hence suffers
from related problems like the uncertainty principle. Moreover, like Fourier,
the wavelet transform too is non-adaptive. The basis functions are selected *a
priori* and consequently make the wavelet decomposition prone to spurious
harmonics and ultimately incorrect interpretations of the data.

A remarkable advantage of Fourier based methods is their mathematical
framework. Fourier based methods are so elegant that they make building
models for a given dataset very easy. Although such models can represent most
of the data and are extensive enough for a practical application, the fact
remains that there is some amount of data slipping through the gaps left
behind by linear approximations. Despite all these shortcomings, wavelet
analysis still remains the best possible method for analysis of nonstationary
data, and hence should be used as a reference to establish the validity
of other methods.

The Uncertainty Principle
-------------------------

A very manifest limitation of the Fourier transform can be seen as the
uncertainty principle. Consider the signal shown here::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> f1, f2 = 500, 1000
    >>> t1, t2 = 0.192, 0.196
    >>> f_sample = 8000
    >>> n_points = 2048
    >>> ts = np.arange(n_points, dtype=float) / f_sample
    >>> signal = np.sin(2 * np.pi * f1 * ts) + np.sin(2 * np.pi * f2 * ts)
    >>> signal[int(t1 * f_sample) - 1] += 3
    >>> signal[int(t2 * f_sample) - 1] += 3
    >>> plt.plot(ts, signal)

.. plot:: tutorials/plots/uncertainty_example_plot.py

It is a sum of two sinusiodal signals of frequencies 500 Hz and 1000 Hz. It has
two spikes at t = 0.192s and t = 0.196s. The purpose of a time frequency
distribution would be to clearly identify both the frequencies and both the spikes,
thus resolving events in both frequency and time. Let's check out the spectrograms of
the STFTs of the signal with four different window lengths:

.. plot:: tutorials/plots/uncertainty_stft.py

As can be clearly seen in Figure 2, the resolution in time and frequency
cannot be obtained simultaneously. In the first (topmost) image, where the
window length is high, the STFT Fig. 2. STFT of signal in Fig 1 with four
different window lengths. Note that the resolution in time and frequency
changes inversely. manages to discriminate between frequencies of 500 Hz and
1000 Hz very clearly, but the time resolution between the events at t = 0.192
s and t = 0.196 s is ambiguous. As we reduce the length of the window function,
the resolution between the time events goes on becoming better, but only at
the cost of resolution in frequencies.

This phenomenon is called the Uncertainty principle. Informally, it states
that arbitrarily high resolution cannot be obtained in both time and frequency.
This is a consequence of the definition of the Fourier transform. The
definition insists that a signal be represented as a weighted sum of sinusoid,
and therefore identifies frequency information that is globally prevalent. As
a workaround to this interpretation, we use the STFT which performs the
Fourier transform on limited periods of the signals. But unfortunately the
period length is defined a priori, thereby showing the uncertainty in either
frequency or time. Mathematically the uncertainty principle is represented
with the Heisenberg-Gabor Inequality [6]:

Definition 1. If T and B are standard deviations of the time
characteristics and bandwidth respectively of a signal s(t),
then
T B ≥ 1/2 (1)

Equation 1 is the Heisenberg-Gabor inequality. It states that the
time-bandwidth product of a signal is lower bounded by 1/2. Gaussian
functions satisfy the equality condition in the equation. A remarkably
insightful commentary on the Uncertainty principle is provided in [7], which
states that the Uncertainty principle is a statement about two variables
whose associated operators do not mutually commute. This helps us apply the
Uncertainty principle in signal processing in the same way as in quantum
physics.

Instantaneous Frequency
-----------------------

As a workaround to the limitations imposed by the Uncertainty principle, we
can define two new measures of signal characteristics called Instantaneous
Frequency and Group Delay. The definition of instantaneous frequency has
remained highly controversial ever since its inception [1], and it is easy to
see why. When something is instantaneous it is localized in time. Since time
and frequency are inverse quantities, localizing frequency in time can be
highly ambiguous. However, a practical definition of instantaneous
frequencies is provided by [6] in the next section.

Analytic Signals and Instantaneous Frequencies
++++++++++++++++++++++++++++++++++++++++++++++

In order to define instantaneous frequencies we must first introduce the
concept of analytic signals. For any real valued signal x(t) we associate a
complex valued signal xa(t) defined as:
xa(t) = x(t) + jxd(t) (2)
where xd(t) is the Hilbert transform of x(t). Then the
instantaneous frequency can be defined as:
f(t) = 1
2π
d
dtargxa(t) (3)

The Hilbert-Huang Transform
+++++++++++++++++++++++++++

The Hilbert-Huang transform (HHT) is a recent technology developed by Norden
E Huang in [1] for analysis of nonlinear and nonstationary phenomena. The
real innovation of the HHT is an iterative algorithm called the Empirical
Mode Decomposition (EMD) which breaks a signal down into so-called Intrinsic
Mode Functions (IMFs) which are characterized by being narrowband and nearly
monocomponent and having a large time-bandwidth product. This allows the IMFs
to have well-defined Hilbert transforms and consequently, physically
meaningful instantaneous frequencies. In the next couple of sections we
briefly describe IMFs and the algorithm, EMD, used to obtain them.

Intrinsic Mode Functions
++++++++++++++++++++++++

Consider the three sinusoidal signals shown in Figure 3. All of them are
identical, except that two of them have a nonzero DC component. Now if we
consider the analytic signals of the sinusoidal waves, all of them will be
circles, but with three different centres, as shown in Figure 4.

Fig. 3. Three sines with different DC components

Fig. 4. Analytical Signals of the Sine Waves

Imagine that each circle is traced out by a rotating phasor centered around
the origin in Figure 4. The angle that the phasor rotates through represents
the instantaneous phase of the signal, and its time differential is the
instantaneous frequency. Figure 5 shows the instantaneous phase and
instantaneous frequencies of the sine waves as per this interpretation. As
shown in the figure, only one sinusoid presents an instantaneous frequency
that is constant and corresponds to the true frequency of the waves. This
wave is the one which has its analytical signal centered around the origin,
thereby allowing the phasor to rotate through a total angle of 2π in one
period. This is the wave that has a zero DC component and is symmetrical
around the time axis.

The fact that true instantaneous frequencies are reproduced only when the
signal is symmetric about the X-axis motivates the definition of an IMF [1].

Definition 2. A function is called an intrinsic mode function
when:
• The number of its extrema and zero-crossings differ at
most by unity.
• The mean of the local envelopes defined by it’s local
maxima and that defined by its local minima should be
zero at all times.

Condition 1 ensures that there are no localized oscillations in the signal
and it crosses the X-axis atleast once before Fig. 5. The top figure shows
the instantaneous phases of the sines in Figure 3 and the bottom figures
shows the instantaneous frequencies. Fig. 6. A nonlinear and a nonstationary
signal to perform EMD upon it goes from one extremum to another. Condition 2
ensures meaningful instantaneous frequencies, as explained in the previous
example. The next section explains the algorithm for extracting IMFs out of a
signal.

Empirical Mode Decomposition
++++++++++++++++++++++++++++

The EMD is an iterative algorithm which breaks a signal down into IMFs. The
process is performed as follows:

1) Find all local extrema in the signal.
2) Join all the local maxima with a cubic spline, creating an upper envelope. Repeat for local minima and create a lower envelope.
3) Calculate the mean of the envelopes.
4) Subtract mean from original signals.
5) Repeat steps 1-4 until result is an IMF.
6) Subtract this IMF from the original signal.
7) Repeat steps 1-6 till there are no more IMFs left in the signal.

To demonstrate this algorithm we consider a noisy, nonlinear and
nonstationary signal showed in figure 6. We extract nine IMFs from the signa
. When there are no more frequency components left to extract, the algorithm
leaves a low-pass residue. The IMFs and the residue are shown in Figure 7.

Properties of Intrinsic Mode Functions
++++++++++++++++++++++++++++++++++++++

By virtue of the EMD algorithm, the decomosition is complete, in that the sum
of the IMFs and the residue subtracted from the input signal leaves behing
only a negligible residue. The decomposition is almost orthogonal, in that
the IMFs are all orthogonal to each Fig. 7. Intrinsic Mode Functions of the
signal in Figure 6 other. Also, as emphasized earlier, the greatest advantage
of the IMFs are well-behaved Hilbert transforms, enabling the extraction of
physically meaningful instantaneous frequencies.

IMFs have large time-bandwidth products, which indicates that they tend to
move away from the lower bound of the Heisenberg-Gabor inequality, thereby
avoiding the limitations of the Uncertainty principle, as explained in
section II(A). The large time-bandwidth product also enables the IMFs to have
group delays that are highly correlated with the instantaneous frequencies.

Two Views of Nonlinear Phenomena
--------------------------------

Despite all its robustness and convenience, the HilbertHuang transform is
unfortunately just an algorithm, without a well-defined mathematical base.
All inferences drawn from it are empirical and can only be corroborated as
such. It lacks the mathematical sophistication of the Fourier framework. On
the plus side it provides a very realistic insight into data.

Thus here we have room for a tradeoff between the mathematical elegance of
the Fourier analysis and the physical significance provided by the
Hilbert-Huang transform. Wavelets are the closest thing to the HHT that not
only have the ability to analyze nonlinear and nonstationary phenomena, but
also a complete mathematical foundation. Unfortunately wavelets are not
adaptive and as such might suffer from problems like Uncertainty principle,
leakages, Gibb’s phenomenon, harmonics, etc - like most of the decomposition
techniques that use a priori basis functions. On the other hand, the basis
functions of the HHT are IMFs which are adaptive and empirical. But EMD is
not a perfect algorithm. For many signals it does not converge down to a set
of finite IMFs. Some experts even believe that there is an inherent
contradiction between the way IMFs are defined and the way EMD is executed.
Thus we can possibly use wavelets as a ’handle’ for the appropriate
extraction of IMFs, and conversely, use IMFs to establish the physical
relevance of wavelet decomposition.

Thus the Hilbert-Huang transform is a alternate view of nonlinear and
nonstationary phenomena, one that is unencumbered by mathematical jargon.
This lack of mathematical sophistication allows researchers to be very
flexible and versatile with its use.

Conclusions
-----------

Consider a dark room with a photosensitive device and a light flashes upon
the device at a given instant. The Fourier interpretation of this phenomenon
would be to consider hundreds (ideally infinitely many) of frequencies which
are in phase exactly at the time when the light is flashed. The frequencies
interfere constructively at that instant to produce the flash of light and
cancel each other out at all the other times. The truth of the matter remains
that there are not so many frequency ’events’ to speak of. But the Fourier
interpretation is mathematically so elegant that sometimes it drives the
physical significance out of the model.

The Hilbert-Huang transform, on the other hand, gives prevalence only to
physically meaningful events. The extraction of instantaneous frequencies
does not depend on convolution (as in the Fourier model), but on time
derivatives. The bases are not chosen a priori, but are adaptive. Table I
shows a detailed comparison of the two ideas. A complementary use of these
two paradigms to analyze nonlinear and nonstationary phenomena has great
research potential.
