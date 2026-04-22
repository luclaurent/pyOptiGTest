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


class FunBukin19(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 19's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=10
        c=0.1
        d=62

        #evaluation and derivatives
        pa=X[:, 1]-X[:, 0]**2+b
        pb=X[:, 1]-X[:, 0]-d
        p=a*np.abs(pa)+c*np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-2*a*X[:, 0]*np.sign(pa)-c*np.sign(pb)
            dp[:, 1]=a*np.sign(pa)+c*np.sign(pb)

        return (p, dp) if grad else p


funBukin19 = FunBukin19()
