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


class FunCigar(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Cigar function
        L. LAURENT -- 14/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-24.15681551650653 for xi=+/- 9.646157266348881
        Design space: -10<xi<10
        """
        #constants
        a=1e6

        #evaluation and derivatives
        x=X[:, 0]
        #
        pa=np.sum(X**2, axis=1)
        #
        p=x**2+a*pa

        if grad:
            #
            dp=np.zeros_like(X)
            #
            dp[:, 0]=2*x+2*a*x
            dp[:, 1:]=2*a*X[:, 1:]

        return (p, dp) if grad else p


funCigar = FunCigar()
