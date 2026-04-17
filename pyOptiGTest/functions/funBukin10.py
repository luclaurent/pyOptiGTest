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


class FunBukin10(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 10's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=20
        c=270
        d=3
        e=30

        #evaluation and derivatives
        pa=X[:, 0]**2+b*np.abs(X[:, 0])+X[:, 1]**2-c
        pb=d*X[:, 0]+X[:, 1]+e
        p=a*pa**2+np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*(2*X[:, 0]+b*np.sign(X[:, 0]))*pa+d*np.sign(pb)
            dp[:, 1]=4*a*X[:, 1]*pa+np.sign(pb)

        return (p, dp) if grad else p


funBukin10 = FunBukin10()
