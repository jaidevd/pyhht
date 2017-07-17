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
        self.sinusoid = np.sin(2 * np.pi * 5 * t)
        self.random_data = np.random.random((1000,))

    def test_get_envelopes(self):
        upper, lower = utils.get_envelops(self.sinusoid)
        self.assertGreaterEqual(np.greater_equal(upper, self.sinusoid).sum(),
                                0.9 * self.sinusoid.shape[0])
        self.assertGreaterEqual(np.less_equal(lower, self.sinusoid).sum(),
                                0.9 * self.sinusoid.shape[0])

    def test_error_not_enough_extrema(self):
        t = np.linspace(0, 1, 1000)
        signal = np.exp(-(t - 500) ** 2)
        self.assertRaises(ValueError, utils.boundary_conditions, signal, t)

    def test_boundary_conditions(self):
        x = np.ones((7,))
        x[[1, 3, 5]] = -1
        x[-1] = 0
        x[0] = 0
        t = np.arange(7)
        tmin, tmax, zmin, zmax = utils.boundary_conditions(x, t, 2)
        self.assertEqual(zmin.sum(), -zmin.shape[0])
        self.assertEqual(zmax.sum(), zmax.shape[0])
        a = np.diff(tmin)
        b = np.diff(tmax)
        np.testing.assert_allclose(a, 2 * np.ones((a.shape[0])))
        np.testing.assert_allclose(b, np.array([2, 4, 2, 4, 2]))

    def test_extrema_sinusoid(self):
        """
        Test if local extrema are detected properly for a trended sinusoid.
        """
        indmin, indmax, _ = utils.extr(self.sinusoid)
        min_neighbours = np.zeros((indmin.shape[0], 2))
        max_neighbours = np.zeros((indmax.shape[0], 2))
        min_neighbours[:, 0] = self.sinusoid[indmin - 1]
        min_neighbours[:, 1] = self.sinusoid[indmin + 1]
        max_neighbours[:, 0] = self.sinusoid[indmax - 1]
        max_neighbours[:, 1] = self.sinusoid[indmax + 1]
        minima = self.sinusoid[indmin].reshape(indmin.shape[0], 1)
        self.assertTrue(np.all(min_neighbours >= minima))
        maxima = self.sinusoid[indmax].reshape(indmax.shape[0], 1)
        self.assertTrue(np.all(max_neighbours <= maxima))

    def test_extrema_random(self):
        """
        Test if local extrema are detected properly for random data.
        """
        indmin, indmax, _ = utils.extr(self.random_data)
        min_neighbours = np.zeros((indmin.shape[0], 2))
        max_neighbours = np.zeros((indmax.shape[0], 2))
        min_neighbours[:, 0] = self.random_data[indmin - 1]
        min_neighbours[:, 1] = self.random_data[indmin + 1]
        max_neighbours[:, 0] = self.random_data[indmax - 1]
        max_neighbours[:, 1] = self.random_data[indmax + 1]
        minima = self.random_data[indmin].reshape(indmin.shape[0], 1)
        self.assertTrue(np.all(min_neighbours >= minima))
        maxima = self.random_data[indmax].reshape(indmax.shape[0], 1)
        self.assertTrue(np.all(max_neighbours <= maxima))

    def test_zerocrossings_sinusoid(self):
        """
        Test if the zero crossings are accurate for a trended sinusoid.
        """
        _, _, indzer = utils.extr(self.sinusoid)
        neighbours = np.zeros((indzer.shape[0], 2))
        neighbours[:, 0] = self.sinusoid[indzer - 1]
        neighbours[:, 1] = self.sinusoid[indzer + 1]
        p = np.prod(neighbours, axis=1)
        self.assertTrue(np.all(p < 0))

    def test_zerocrossings_random(self):
        """
        Test if the zero crossings are accurate for a trended sinusoid.
        """
        _, _, indzer = utils.extr(self.random_data)
        neighbours = np.zeros((indzer.shape[0], 2))
        neighbours[:, 0] = self.random_data[indzer - 1]
        neighbours[:, 1] = self.random_data[indzer + 1]
        p = np.prod(neighbours, axis=1)
        self.assertTrue(np.all(p < 0))


if __name__ == '__main__':
    unittest.main()
