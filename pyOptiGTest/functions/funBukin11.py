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


class FunBukin11(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 11's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=5

        #evaluation and derivatives
        pa=X[:, 0]+b
        pb=X[:, 1]+b
        p=a*np.sin(X[:, 0]-X[:, 1])**2+np.abs(pa)+np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*np.cos(X[:, 0]-X[:, 1])*np.sin(X[:, 0]-X[:, 1])+2*pa
            dp[:, 1]=-2*a*np.cos(X[:, 0]-X[:, 1])*np.sin(X[:, 0]-X[:, 1])+2*pb

        return (p, dp) if grad else p


funBukin11 = FunBukin11()
