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


class FunBukin09(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 9's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=5
        c=9

        #evaluation and derivatives
        pa=X[:, 0]-b*X[:, 1]-X[:, 1]**2
        pb=X[:, 1]+X[:, 0]+c
        p=a*pa**2+np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*pa+np.sign(pb)
            dp[:, 1]=2*a*(-b-2*X[:, 1])*pa+np.sign(pb)

        return (p, dp) if grad else p


funBukin09 = FunBukin09()
