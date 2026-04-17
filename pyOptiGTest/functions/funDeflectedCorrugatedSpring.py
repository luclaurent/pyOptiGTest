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


class FunDeflectedCorrugatedSpring(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for xi=alpha
        Design space: 0<xi<2*alpha
        alpha=5;
        """
        #constants
        K=5
        alpha=5
        a=0.1

        #evaluation and derivatives
        xa=X-alpha
        #
        sxa=np.sum(xa**2, axis=1)
        sqxa=np.sqrt(sxa)
        #
        p=a*sxa-np.cos(K*sqxa)
        #

        #keyboard
        #
        if grad:
            #
            dp=2*a*xa+K*xa/sqxa*np.sin(K*sqxa)

        return (p, dp) if grad else p


funDeflectedCorrugatedSpring = FunDeflectedCorrugatedSpring()
