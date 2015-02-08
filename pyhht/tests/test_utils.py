#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
Tests for the basic utility functions in `pyhht.utils`
"""

import unittest
from pyhht import utils
import numpy as np
from numpy.testing import assert_allclose


class TestUtils(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0, 1, 1000)
        self.data = np.sin(2 * np.pi * 5 * t)

    def test_extr_zerocross(self):
        """
        Test if local extrema and zero crossings are detected properly.
        """
        # FIXME: Don't use the fact that the data is a sine wave.
        # Eg: Use the properties of immediate neighbours
        # FIXME: Test zero crossings more carefully, the rest is just
        # `scipy.argrelextrema`
        indmin, indmax, indzer = utils.extr(self.data)
        desired = np.ones((5,))
        actual = self.data[indmax]
        assert_allclose(actual, desired)
        desired *= -1
        actual = self.data[indmin]
        assert_allclose(actual, desired)
