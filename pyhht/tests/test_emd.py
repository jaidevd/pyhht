#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Unittests for the EMD class
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose
from pyhht.emd import EMD


class TestEMD(unittest.TestCase):

    def setUp(self):
        self.trend = self.ts = np.linspace(0, 1, 10000)
        self.mode1 = np.sin(2 * np.pi * 5 * self.ts)
        self.mode2 = np.sin(2 * np.pi * 10 * self.ts)

    def test_imfs_total_no_error(self):
        """
        Check if the sum of the IMFs is sufficiently close to the input signal.
        """
        signal = np.sum([self.trend, self.mode1, self.mode2], axis=0)
        emd = EMD(signal)
        imfs = emd.decompose()
        assert_allclose(imfs.sum(0), signal)

    def test_monotonicity_of_trend(self):
        """
        Check if the trend is monotonic.
        """
        signal = np.sum([self.trend, self.mode1, self.mode2], axis=0)
        emd = EMD(signal)
        imfs = emd.decompose()
        # There should be two IMFs, and the rest of them are trends
        trend = imfs[3:, :].sum(0)
        assert_allclose(self.trend, trend)

if __name__ == '__main__':
    unittest.main()
