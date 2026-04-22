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


class FunMishra10(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 10 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for x=(2,2)
        Design space: -10<xi<10
        """

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        fx=np.floor(xxx)
        fy=np.floor(yyy)
        p=(fx*fy-fx-fy)**2

        if grad:
            dp=np.full(X.shape, np.nan)

        return (p, dp) if grad else p


funMishra10 = FunMishra10()
