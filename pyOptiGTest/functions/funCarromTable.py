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


class FunCarromTable(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 14/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-24.15681551650653 for xi=+/- 9.646157266348881
        Design space: -10<xi<10
        """
        #constants
        a=30
        b=2
        c=1
        d=np.pi

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        xy=np.sqrt(x**2+y**2)
        fxy=c-xy/d
        #
        pa=np.exp(b*np.abs(fxy))
        cx=np.cos(x)
        cy=np.cos(y)
        #
        p=-1/a*pa*cx**2*cy**2

        if grad:
            #
            dp=-b/d*X/xy[:, None]*np.sign(fxy)[:, None]*p[:, None] +2*np.sin(X)*np.cos(X)/a*pa[:, None]*np.cos(X[:, [1, 0]])**2

        return (p, dp) if grad else p


funCarromTable = FunCarromTable()
