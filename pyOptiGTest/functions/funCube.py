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


class FunCube(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 05/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        4 global minimas : f(x1,x2,x3,x4)=0 for (-1,1)
        Design space: -10<xi<10
        """

        #constants
        a=100
        b=1

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=yyy-xxx**3
        pb=b-xxx
        p=a*pa**2+pb**2
        #
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-6*a*xxx**2*pa-2*pb
            dp[:, 1]=2*a*pa

        return (p, dp) if grad else p


funCube = FunCube()
