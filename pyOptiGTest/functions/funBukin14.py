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


class FunBukin14(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 14's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=1e-3
        c=11

        #evaluation and derivatives
        pa=X[:, 1]-b*X[:, 0]**3
        pb=X[:, 1]+X[:, 0]+c
        p=a*np.abs(pa)+np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-3*a*b*X[:, 0]**2*np.sign(pa)+np.sign(pb)
            dp[:, 1]=a*np.sign(pa)+np.sign(pb)

        return (p, dp) if grad else p


funBukin14 = FunBukin14()
