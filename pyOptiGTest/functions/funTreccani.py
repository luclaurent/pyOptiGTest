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


class FunTreccani(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 23/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(-2,0)=0
        Design space -5<xi<5
        """

        #constants
        a=4

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        p=x**4+a*x**3+a*x**2+y**2

        if grad:
            #
            dp=np.zeros_like(X)
            #
            dp[:, 0]=4*x**3+3*a*x**2+2*a*x
            dp[:, 1]=2*y

        return (p, dp) if grad else p


funTreccani = FunTreccani()
