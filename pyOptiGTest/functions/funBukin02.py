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


class FunBukin02(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 2's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=0.01
        c=1
        d=10

        #evaluation and derivatives
        p=a*(X[:, 1]-b*X[:, 0]**2+c)**2+b*(X[:, 0]+d)**2
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*b*(X[:, 0]+d)-a*b*4*X[:, 0]*(X[:, 1]-b*X[:, 0]**2+c)
            dp[:, 1]=2*a*(X[:, 1]-b*X[:, 0]**2+c)

        return (p, dp) if grad else p


funBukin02 = FunBukin02()
