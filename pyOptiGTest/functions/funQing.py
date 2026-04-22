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


class FunQing(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Qing function
        L. LAURENT -- 19/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for xi=(+/-sqrt(i),...,+/-sqrt(i))
        Design space: -500<xi<500
        """

        #evaluation and derivatives
        dim=X.shape[1]
        pa=(X**2) - (np.arange(1, dim+1))
        #
        p=np.sum(pa**2, axis=1)

        if grad:
            dp=4*X*pa

        return (p, dp) if grad else p


funQing = FunQing()
