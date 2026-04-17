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


class FunBukin04(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 4's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=0.01
        c=10

        #evaluation and derivatives
        pa=X[:, 0]+c
        p=a*X[:, 1]**2+b*np.abs(pa)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=b*np.sign(pa)
            dp[:, 1]=2*a*X[:, 1]

        return (p, dp) if grad else p


funBukin04 = FunBukin04()
