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


class FunBukin05(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 5's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=10
        c=0.01

        #evaluation and derivatives
        pa=X[:, 0]+b
        p=c*X[:, 1]**2+a*np.abs(pa)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=a*np.sign(pa)
            dp[:, 1]=2*c*X[:, 1]

        return (p, dp) if grad else p


funBukin05 = FunBukin05()
