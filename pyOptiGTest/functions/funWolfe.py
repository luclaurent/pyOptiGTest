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


class FunWolfe(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Wolfe function
        L. LAURENT -- 28/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,0,0)=0
        Design space 0<xi<2
        """

        #constants
        a=14/3
        b=0.75

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        z=X[:, 2]
        #
        p=a*(x**2+y**2-x*y)**b+z

        if grad:
            dp=np.ones_like(X)
            pb=(x**2+y**2-x*y)**(b-1)
            #
            dp[:, 0]=a*b*(2*x-y)*pb
            dp[:, 1]=a*b*(2*y-x)*pb

        return (p, dp) if grad else p


funWolfe = FunWolfe()
