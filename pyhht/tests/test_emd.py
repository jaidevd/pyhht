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
from scipy.signal import argrelmax, argrelmin
from numpy.testing import assert_allclose
from pyhht.emd import EMD


class TestEMD(unittest.TestCase):

    def setUp(self):
        self.trend = self.ts = np.linspace(0, 1, 10000)
        self.mode1 = np.sin(2 * np.pi * 5 * self.ts)
        self.mode2 = np.sin(2 * np.pi * 10 * self.ts)

    def test_emd_multidimensional_signal_error(self):
        """Check if EMD raises an error for multidimensional signals."""
        signal = self.trend + self.mode1 + self.mode2
        signal = np.c_[signal, signal]
        self.assertRaises(ValueError, EMD, signal)

    def test_imfs_total_no_error(self):
        """
        Check if the sum of the IMFs is sufficiently close to the input signal.
        """
        signal = np.sum([self.trend, self.mode1, self.mode2], axis=0)
        emd = EMD(signal)
        imfs = emd.decompose()
        assert_allclose(imfs.sum(0), signal)

    def test_decomposition(self):
        """Test the decompose method of the emd class."""
        signal = np.sum([self.trend, self.mode1, self.mode2], axis=0)
        decomposer = EMD(signal, t=self.ts)
        imfs = decomposer.decompose()
        self.assertItemsEqual(imfs.shape, (signal.shape[0], 3))

    def test_residue(self):
        """Test the residue of the emd output."""
        signal = np.sum([self.trend, self.mode1, self.mode2], axis=0)
        decomposer = EMD(signal, t=self.ts)
        imfs = decomposer.decompose()
        n_imfs = imfs.shape[0]
        n_maxima = argrelmax(imfs[n_imfs - 1, :])[0].shape[0]
        n_minima = argrelmin(imfs[n_imfs - 1, :])[0].shape[0]
        self.assertTrue(max(n_maxima, n_minima) <= 2)


if __name__ == '__main__':
    unittest.main()
