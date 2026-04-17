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


class FunBukin20(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 20's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=5
        c=9
        d=0.1
        e=4
        f=6

        #evaluation and derivatives
        pa=X[:, 1]-b*X[:, 0]-c
        pb=e*X[:, 1]+X[:, 0]+f
        p=a*pa**2+d*pb**2
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-2*a*b*pa+2*d*pb
            dp[:, 1]=2*a*pa+2*d*e*pb

        return (p, dp) if grad else p


funBukin20 = FunBukin20()
