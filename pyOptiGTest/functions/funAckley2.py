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


class FunAckley2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Ackley's function 1
        L. LAURENT -- 31/10/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        1 global minimum : x=(0,0,...,0) >> f(x)=0
        design space -35<xi<35 (small range -2<xi<2)
        Ackley's function 2
        L. LAURENT -- 31/10/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        one local minimum
        1 global minimum : x=(0,0) >> f(x)=-200
        design space -32<xi<32
        """

        #constants
        a=200
        b=0.02

        #responses and derivatives
        normP=np.sqrt(np.sum(X**2, axis=1))
        ex1=np.exp(-b*normP)
        p=-a*ex1

        if grad:
            dp=a*b*X/normP[:, None]*ex1[:, None]

        return (p, dp) if grad else p


funAckley2 = FunAckley2()
