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


class FunSchwefel36(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 36 function
        L. LAURENT -- 28/04/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        1 global minimum : x=(12,12) >> f(x)=-3456
        Design space: 0<xi<10
        """

        #constants
        a=72
        b=2

        #evaluation and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]
        #
        pa=xxx*yyy
        pb=a-b*xxx-b*yyy
        #
        p=-pa*pb
        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            dp[:, 0]=-yyy*pb+b*pa
            dp[:, 1]=-xxx*pb+b*pa


        return (p, dp) if grad else p


funSchwefel36 = FunSchwefel36()
