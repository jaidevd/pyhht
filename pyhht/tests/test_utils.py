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


class TestUtils(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0, 1, 1000)
        self.data = np.sin(2 * np.pi * 5 * t)

    def test_extrema(self):
        """
        Test if local extrema are detected properly.
        """
        # FIXME: Try tests on random data.
        indmin, indmax, _ = utils.extr(self.data)
        min_neighbours = np.zeros((indmin.shape[0], 2))
        max_neighbours = np.zeros((indmax.shape[0], 2))
        min_neighbours[:, 0] = self.data[indmin - 1]
        min_neighbours[:, 1] = self.data[indmin + 1]
        max_neighbours[:, 0] = self.data[indmax - 1]
        max_neighbours[:, 1] = self.data[indmax + 1]
        minima = self.data[indmin].reshape(indmin.shape[0], 1)
        self.assertTrue(np.all(min_neighbours >= minima))
        maxima = self.data[indmax].reshape(indmax.shape[0], 1)
        self.assertTrue(np.all(max_neighbours <= maxima))

    def test_zerocrossings(self):
        """
        Test if the zero crossings are accurate.
        """
        _, _, indzer = utils.extr(self.data)
        neighbours = np.zeros((indzer.shape[0], 2))
        neighbours[:, 0] = self.data[indzer - 1]
        neighbours[:, 1] = self.data[indzer + 1]
        p = np.prod(neighbours, axis=1)
        self.assertTrue(np.all(p < 0))
