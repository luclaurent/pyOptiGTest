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


class FunChungReynolds(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 04/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        1 global minimum : x=(0,0,...,0) >> f(x)=0
        design space -1<xi<4
        """


        #evaluation and derivatives
        pSum=np.sum(X**2, axis=1)
        p=pSum**2
        if grad:
            dp=4*X*pSum[:, None]

        return (p, dp) if grad else p


funChungReynolds = FunChungReynolds()
