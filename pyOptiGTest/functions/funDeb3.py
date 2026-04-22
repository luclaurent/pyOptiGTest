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


class FunDeb3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Deb3's function
        L. LAURENT -- 06/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        5^np global minimum :
        design space -1<xi<1
        """

        #constans
        a=6
        b=5*np.pi
        c=3/4
        d=0.05

        #evaluation and derivatives
        nS=X.shape[1]
        sx=np.sin(b*(X**c-d))
        p=-1/nS*np.sum(sx**a, axis=1)

        if grad:
            cx=np.cos(b*(X**c-d))
            dp=-1/nS*b*a*c*X**(c-1)*cx*sx**(a-1)

        return (p, dp) if grad else p


funDeb3 = FunDeb3()
