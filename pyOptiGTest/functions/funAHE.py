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


class FunAHE(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Axis parallel hyper-ellipsoid (Weighted Sphere Model) function
        L. LAURENT -- 21/02/2012 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 global minimum: x=(0,0,...,0) >> f(x)=0
        design space -5.12<xi<5.12
        """

        #number of design variables
        nbvar=X.shape[1]

        # responses and derivatives
        vv=np.arange(1, nbvar+1)
        coef=vv
        cal=coef*X**2
        p=np.sum(cal, axis=1)
        if grad:
            dp=2*coef*X

        return (p, dp) if grad else p


funAHE = FunAHE()
