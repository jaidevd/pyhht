import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import argrelmax
import matplotlib.pyplot as plt


def loaddata():
    data = loadmat('docs/examples/datasets/eastern_basin.mat')
    X = data['X'].ravel()
    Y = data['Y'].ravel()
    return (X + 1j * Y) / 1000


def plot_complex(x):
    plt.plot(np.real(x), 'b')
    plt.plot(np.imag(x), 'k--')


def bivariate_sift(signal, phi):
    projections = np.real(np.exp(-1j * phi) * signal.reshape(-1, 1))
    envelopes = np.zeros((phi.shape[0], signal.shape[0]), dtype=complex)
    for i, direction in enumerate(range(phi.shape[0])):
        projection = projections[:, direction]
        maxima_loc = argrelmax(projection)[0]
        maxima_expand = np.zeros((maxima_loc.shape[0] + 2,), dtype=int)
        maxima_expand[1:-1] = maxima_loc
        maxima_expand[-1] = signal.shape[0] - 1
        maxima_val = (np.exp(1j * phi[direction]) * projection)[maxima_expand]
        interpolant = interp1d(maxima_expand, maxima_val)
        envelope = interpolant(np.arange(signal.shape[0]))
        envelopes[i] = envelope
    m_envelope = envelopes.sum(0) * 2 / phi.shape[0]
    return m_envelope


if __name__ == '__main__':
    signal = loaddata()
    # n_directions = 64
    # phi = np.arange(1, n_directions + 1) * 2 * np.pi / n_directions
    # sift_iter = 10
    # n_imfs = 4
    # imfs = np.zeros((n_imfs, signal.shape[0]), dtype=complex)
    # for j in range(n_imfs):
        # residue = signal - imfs.sum(0)
        # for i in range(sift_iter):
            # env = bivariate_sift(residue, phi)
            # residue -= env
        # imfs[j] = residue.copy()
    #
    # plt.subplot(n_imfs + 2, 1, 1)
    # plot_complex(signal)
    # plt.xticks([])
    # for i in range(n_imfs):
        # plt.subplot(n_imfs + 2, 1, i + 2)
        # plot_complex(imfs[i])
        # plt.xticks([])
    # trend = signal - imfs.sum(0)
    # plt.subplot(n_imfs + 2, 1, n_imfs + 2)
    # plot_complex(trend)
    # plt.xticks([])
    # plt.tight_layout()
    # plt.show()
    from pyhht import EMD
    from pyhht.visualization import plot_imfs
    emd = EMD(signal, n_imfs=4)
    imfs = emd.decompose()
    plot_imfs(signal, imfs)
