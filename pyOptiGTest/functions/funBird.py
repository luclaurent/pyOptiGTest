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


class FunBird(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bird's function
        L. LAURENT -- 31/10/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        one local minimum
        1 global minimum : x={(4.70104,3.15294),(-1.58214, -3.13024)} >> f(x)=-106.764537
        design space -2pi<xi<2pi
        """

        #evaluate responses
        xxx=X[:, 0]
        yyy=X[:, 1]
        sx=np.sin(xxx)
        cy=np.cos(yyy)
        ex1=np.exp((1-cy)**2)
        ex2=np.exp((1-sx)**2)
        fxy=(xxx-yyy)**2
        p=sx*ex1+cy*ex2+fxy

        #evaluate derivatives
        if grad:
            cx=np.cos(xxx)
            sy=np.sin(yyy)
            dp = np.zeros_like(X)
            dp[:, 0]=cx*ex1-2*cy*cx*(1-sx)*ex2+2*(xxx-yyy)
            dp[:, 1]=2*sx*sy*(1-cy)*ex1-sy*ex2-2*(xxx-yyy)

        return (p, dp) if grad else p


funBird = FunBird()
