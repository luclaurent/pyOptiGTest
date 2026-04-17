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


class FunBukin08(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 8's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=800
        c=40

        #evaluation and derivatives
        pa=X[:, 1]**2+X[:, 0]**2-b
        pb=X[:, 1]+X[:, 0]+c
        p=a*np.abs(pa)+np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*X[:, 0]*np.sign(pa)+np.sign(pb)
            dp[:, 1]=2*a*X[:, 1]*np.sign(pa)+np.sign(pb)

        return (p, dp) if grad else p


funBukin08 = FunBukin08()
