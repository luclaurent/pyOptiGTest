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


class FunAckley3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Ackley's function 3
        L. LAURENT -- 31/10/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        one local minimum
        1 global minimum : x=(0,-0.4) >> f(x)=-219.1418
        design space -32<xi<32
        """

        #constants
        a=200
        b=0.02
        c=5
        d=3

        #responses and derivatives
        normP=np.sqrt(np.sum(X**2, axis=1))
        ex1=np.exp(-b*normP)
        ex2=np.exp(np.cos(d*X[:, 0])+np.sin(d*X[:, 1]))
        p=-a*ex1-c*ex2

        if grad:
            dp=a*b*X/normP[:, None]*ex1[:, None]
            #
            dp[:, 0]=dp[:, 0]+c*d*np.sin(d*X[:, 0])*ex2
            dp[:, 1]=dp[:, 1]-c*d*np.cos(d*X[:, 1])*ex2

        return (p, dp) if grad else p


funAckley3 = FunAckley3()
