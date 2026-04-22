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


class FunMishra03(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 3 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-0.18467 for x=(-8.4667,-10)
        Design space: -10<xi<10
        """

        #constants
        a=1e-2

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=xxx**2+yyy
        pb=xxx+yyy
        #
        pc=np.abs(pa)
        pd=np.sqrt(pc)
        #
        p=np.sqrt(np.abs(np.cos(pd)))+a*pb

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-0.5*xxx/pd *np.sin(pd) *np.sign(np.cos(pd)) *np.sign(pa) /np.sqrt(np.abs(np.cos(pd))) +a
            dp[:, 1]=-0.25/pd *np.sin(pd) *np.sign(np.cos(pd)) *np.sign(pa) /np.sqrt(np.abs(np.cos(pd))) +a

        return (p, dp) if grad else p


funMishra03 = FunMishra03()
