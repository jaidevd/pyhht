import os.path as op
from numpy import pi, sin, linspace
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.testing.decorators import image_comparison  # noqa: F402
from pyhht.emd import EMD  # noqa: F402
from pyhht.visualization import plot_imfs  # noqa: F402


@image_comparison(baseline_images=['imfs'], extensions=['png'])
def test_plot_imfs():
    """Test if plotting IMFs works correctly."""
    t = linspace(0, 1, 1000)
    modes = sin(2 * pi * 5 * t) + sin(2 * pi * 10 * t)
    x = modes + t
    decomposer = EMD(x)
    imfs = decomposer.decompose()
    plot_imfs(x, imfs, t, show=False)


@image_comparison(baseline_images=['imf_no_timestamp'], extensions=['png'])
def test_plot_imfs_no_ts():
    """Test if plotting IMFs works when no timestamp is provided."""
    t = linspace(0, 1, 1000)
    modes = sin(2 * pi * 5 * t) + sin(2 * pi * 10 * t)
    x = modes + t
    decomposer = EMD(x)
    imfs = decomposer.decompose()
    plot_imfs(x, imfs, show=False)


@image_comparison(baseline_images=['bivariate_emd'], extensions=['png'])
def test_plot_bivariate():
    """Test if plotting bivariate IMFs works."""
    dataset_path = op.join(op.dirname(__file__), '..', '..', 'docs',
                           'examples', 'datasets', 'eastern_basin.mat')
    data = loadmat(dataset_path)
    signal = data['X'].ravel() + 1j * data['Y'].ravel()
    decomposer = EMD(signal)
    imfs = decomposer.decompose()
    plot_imfs(signal, imfs, show=False)
