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


class FunPaviani(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Paviani function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-45.778 for x=[9.351,...,9.351]
        Design space: 2.0001<xi<10
        """

        #constants
        a=2
        b=10
        c=0.2

        #evaluation and derivatives
        dim=X.shape[1]
        xa=X-a
        xb=b-X
        la=np.log(xa)
        lb=np.log(xb)
        #
        pa=np.sum(la**2+lb**2, axis=1)
        pb=np.prod(X, axis=1)
        p=pa-pb**c

        if grad:
            # product-without-i: pd[:, i] = prod(X) / X[:, i]
            pd = pb[:, None] / X
            #
            dp=2*la/xa-2*lb/xb-c*pb[:, None]**(c-1) * pd

        return (p, dp) if grad else p


funPaviani = FunPaviani()
