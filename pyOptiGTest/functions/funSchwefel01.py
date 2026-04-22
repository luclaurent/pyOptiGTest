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


class FunSchwefel01(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 1 function
        L. LAURENT -- 19/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -100<xi<100
        """

        #constants
        a=np.sqrt(np.pi)

        #evaluation and derivatives
        pa=X**2
        spa=np.sum(pa, axis=1)
        #
        p=spa**a

        if grad:
            #
            dp=2*a*X*spa[:, None]**(a-1)

        return (p, dp) if grad else p


funSchwefel01 = FunSchwefel01()
