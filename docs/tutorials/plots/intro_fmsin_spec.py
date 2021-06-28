from tftb.generators import fmsin
from tftb.processing import ShortTimeFourierTransform
sig, iflaw = fmsin(256, 0.1, 0.3, period=64)

stft = ShortTimeFourierTransform(sig)
stft.run()
stft.plot()
