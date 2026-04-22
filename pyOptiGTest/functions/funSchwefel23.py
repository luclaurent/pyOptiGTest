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


class FunSchwefel23(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 23 function
        L. LAURENT -- 28/04/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        1 global minimum : x=(0,0,...,0) >> f(x)=0
        Design space: -10<xi<10
        """

        #constants
        a=10

        #evaluation and derivatives
        pa=X**a
        #
        p=np.sum(pa, axis=1)
        #
        if grad:
            #
            dp=a*X**(a-1)


        return (p, dp) if grad else p


funSchwefel23 = FunSchwefel23()
