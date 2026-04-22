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


class FunPrice01(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for x={(?5,?5),(?5,5),(5,?5),(5,5)}
        Design space: -500<xi<500
        """

        #constants
        a=5

        #evaluation and derivatives
        #
        xa=np.abs(X)-a
        p=np.sum(xa**2, axis=1)

        if grad:
            dp=2*np.sign(X)*xa

        return (p, dp) if grad else p


funPrice01 = FunPrice01()
