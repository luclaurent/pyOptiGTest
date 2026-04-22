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


class FunSchwefel20(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 20 function
        L. LAURENT -- 19/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -100<xi<100
        """

        #evaluation and derivatives
        pa=np.abs(X)
        #
        p=np.sum(pa, axis=1)

        if grad:
            #
            dp=np.sign(X)

        return (p, dp) if grad else p


funSchwefel20 = FunSchwefel20()
