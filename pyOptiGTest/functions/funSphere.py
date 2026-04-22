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


class FunSphere(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Sphere function
        L. LAURENT -- 16/09/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(xi)=0 pour (x1,x2,x3,x4)=(0,...,0)
        Design space: -1<xi<1
        """

        #evaluation and derivatives
        pX=X**2
        #
        p=np.sum(pX, axis=1)

        if grad:
            #
            dp=2*X

        return (p, dp) if grad else p


funSphere = FunSphere()
