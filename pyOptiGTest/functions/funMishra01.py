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


class FunMishra01(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 1 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=2 for x=?
        Design space: 0<xi<1
        """
        #constants
        a=1

        #evaluation and derivatives
        dim=X.shape[1]
        xn=dim-np.sum(X, axis=1)
        #
        p=(a+xn)**xn

        if grad:
            dp=((-np.log(a+xn)-xn/(a+xn))*p)[:, None] * np.ones((1, dim))

        return (p, dp) if grad else p


funMishra01 = FunMishra01()
