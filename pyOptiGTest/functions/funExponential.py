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


class FunExponential(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Exponential function
        L. LAURENT -- 16/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        one local minimum
        1 global minimum : x=(0,...,0) >> f(x)=1
        design space -1<xi<1
        """

        #constants
        a=0.5

        #evaluations and derivatives
        p=-np.exp(-a*np.sum(X**2, axis=1))

        if grad:
            dp=-2*a*(p) * (X)

        return (p, dp) if grad else p


funExponential = FunExponential()
