import numpy as np
from tftb.generators import fmconst
from tftb.processing import ShortTimeFourierTransform
n_points = 128
mode1, iflaw1 = fmconst(n_points, fnorm=0.1)
mode2, iflaw2 = fmconst(n_points, fnorm=0.3)
signal = np.r_[mode1, mode2]

stft = ShortTimeFourierTransform(signal)
stft.run()
stft.plot()
