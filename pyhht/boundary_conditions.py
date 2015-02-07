import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
from scipy.interpolate import splrep, splev


def boundary_conditions(x, t=None):
    
    """ Generates mirrored extrema beyond the singal limits. """
    return

Fs = 10000.0
ts = np.linspace(0, 1, Fs)
f1, f2 = 5, 10
y1 = np.sin(2*np.pi*f1*ts)
y2 = np.sin(2*np.pi*f2*ts)
y = y1 + y2


# Maxima
maxima = argrelmax(y)[0]
ymax = y[maxima]

left = ymax[:2][::-1]
right = ymax[-2:][::-1]
ymax_ext = np.hstack((left, ymax, right))

tmax = ts[maxima]
left = ts.min() - tmax[:2][::-1]
right = 2 * ts.max() - tmax[-2:][::-1]
tmax_ext = np.hstack((left, tmax, right))


# Minima
minima = argrelmin(y)[0]
ymin = y[minima]

left = ymin[:2][::-1]
right = ymin[-2:][::-1]
ymin_ext = np.hstack((left, ymin, right))

tmin = ts[minima]
left = ts.min() - tmin[:2][::-1]
right = 2 * ts.max() - tmin[-2:][::-1]
tmin_ext = np.hstack((left, tmin, right))


# Interpolation
tck = splrep(tmax_ext, ymax_ext)
N = ts.shape[0] + Fs*(np.abs(tmax_ext[0] - ts[0]) + np.abs(tmax_ext[-1] - ts[
    -1]))
_t_top = np.linspace(tmax_ext[0], tmax_ext[-1], np.round(N))
top_env = splev(_t_top, tck)

tck = splrep(tmin_ext, ymin_ext)
N = ts.shape[0] + Fs*(np.abs(tmin_ext[0] - ts[0]) + np.abs(tmin_ext[-1] - ts[
    -1]))
_t_bot = np.linspace(tmin_ext[0], tmin_ext[-1], np.round(N))
bot_env = splev(_t_bot, tck)

plt.plot(ts, y, 'b', tmax_ext, ymax_ext, 'r.', tmin_ext, ymin_ext, 'g.')
plt.plot(_t_top, top_env, 'r-')
plt.plot(_t_bot, bot_env, 'g-')
plt.show()
