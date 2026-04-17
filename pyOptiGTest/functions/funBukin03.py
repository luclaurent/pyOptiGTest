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


class FunBukin03(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 3's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=1.5*np.pi

        #evaluation and derivatives
        pa=(X[:, 1]-np.cos(X[:, 0]))
        pb=X[:, 1]-X[:, 0]-b
        p=(a*pa**2+pb**2)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*np.sin(X[:, 0])*pa-2*pb
            dp[:, 1]=2*a*pa+2*pb

        return (p, dp) if grad else p


funBukin03 = FunBukin03()
