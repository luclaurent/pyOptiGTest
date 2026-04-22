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


class FunPeaks(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Peaks function
        L. LAURENT -- 12/05/2010 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        """
        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        p =  3*(1-xxx)**2*np.exp(-(xxx**2) - (yyy+1)**2) - 10*(xxx/5 - xxx**3 - yyy**5)*np.exp(-xxx**2-yyy**2) - 1/3*np.exp(-(xxx+1)**2 - yyy**2)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-6*(1-xxx)*np.exp(-(xxx**2) - (yyy+1)**2) -6*xxx*(1-xxx)**2*np.exp(-xxx**2-(yyy+1)**2) -10*(1/5-3*xxx**2)*np.exp(-xxx**2-yyy**2) +20*(xxx/5-xxx**3-yyy**5)*xxx*np.exp(-xxx**2-yyy**2) +2/3*(xxx+1)*np.exp(-(xxx+1)**2-yyy**2)
            dp[:, 1]=-6*(1-xxx)**2*(yyy+1)*np.exp(-xxx**2-(yyy+1)**2) +50*yyy**4*np.exp(-xxx**2-yyy**2) +20*yyy*(xxx/5-xxx**3-yyy**5)*np.exp(-xxx**2 -yyy**2) +2/3*yyy*np.exp(-(xxx+1)**2-yyy**2)

        return (p, dp) if grad else p


funPeaks = FunPeaks()
