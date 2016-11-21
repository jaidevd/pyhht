#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""

"""


from numpy import pi, sin, linspace
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

t = linspace(0, 1, 1000)
modes = sin(2 * pi * 5 * t) + sin(2 * pi * 10 * t)
x = modes + t
decomposer = EMD(x)
imfs = decomposer.decompose()

plot_imfs(x, imfs, t)
