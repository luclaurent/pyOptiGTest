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


class FunSalomon(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Salomon function
        L. LAURENT -- 14/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -100<xi<100
        """

        #constants
        a=1
        b=2*np.pi
        c=0.1

        #evaluation and derivatives
        sumx=np.sum(X**2, axis=1)
        sqx=np.sqrt(sumx)
        #
        p=a-np.cos(b*sqx)+c*sqx

        if grad:
            #
            ds=X/sqx
            dp=ds*(b*np.sin(b*sqx)+c)

        return (p, dp) if grad else p


funSalomon = FunSalomon()
