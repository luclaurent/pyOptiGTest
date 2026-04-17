"""
    pyOptiGTest - Python implementation of optimization test functions.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

import numpy as np

from pyOptiGTest.base import TestFunction


class FunCola(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Cola function
        17 variables positioning 10 molecules in 2D
        Design space: 0<x1<4
        """
        dd = np.array([
            [0,    0,    0,    0,    0,    0,    0,    0,    0],
            [1.27, 0,    0,    0,    0,    0,    0,    0,    0],
            [1.69, 1.43, 0,    0,    0,    0,    0,    0,    0],
            [2.04, 2.35, 2.45, 0,    0,    0,    0,    0,    0],
            [3.09, 3.18, 3.26, 2.85, 0,    0,    0,    0,    0],
            [3.20, 3.22, 3.27, 2.88, 1.55, 0,    0,    0,    0],
            [2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0,    0,    0],
            [3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00, 0,    0],
            [3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32, 0],
            [2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97]])

        n_samples = X.shape[0]
        n_vars = X.shape[1]

        # Build coordinate arrays for molecules
        # MATLAB: xC = cat(3, zeros, xx(:,:,1:2:end))  -> 1 + ceil(n_vars/2) cols
        # MATLAB: yC = cat(3, zeros, zeros, xx(:,:,2:2:end)) -> 2 + floor(n_vars/2) cols
        x_odd = X[:, 0::2]   # cols 0,2,4,... -> n_xodd = ceil(n_vars/2)
        y_even = X[:, 1::2]  # cols 1,3,5,... -> n_yeven = floor(n_vars/2)

        n_pts = 1 + x_odd.shape[1]  # first point is at origin
        xC = np.zeros((n_samples, n_pts))
        yC = np.zeros((n_samples, n_pts))
        xC[:, 1:] = x_odd
        yC[:, 2:2 + y_even.shape[1]] = y_even

        p = np.zeros(n_samples)
        for itI in range(n_pts):
            for itJ in range(itI):
                if itI < dd.shape[0] and itJ < dd.shape[1]:
                    dist = np.sqrt((xC[:, itI] - xC[:, itJ])**2 + (yC[:, itI] - yC[:, itJ])**2)
                    p = p + (dist - dd[itI, itJ])**2

        if grad:
            dp = np.zeros_like(X)

        return (p, dp) if grad else p



funCola = FunCola()
