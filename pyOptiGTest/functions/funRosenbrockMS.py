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


class FunRosenbrockMS(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Modified Rosenbrock function (Sobester 2005)
        L. LAURENT -- 16/05/2012 -- luc.laurent@lecnam.net
        A. S\'obester, S. J. Leary, and A. J. Keane. On the design of optimization strategies based on global response surface approximation models. Journal of Global Optimization, 33(1):31?59, 2005.
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=1
        c=75
        d=5

        #evaluation and derivatives
        pa=X[:, 1:]-X[:, :-1]**2
        pb=X[:, :-1]-b
        calA=a*pa**2+pb**2
        sd=np.sin(d*(1-X))
        calB=c*sd
        #
        p=np.sum(calA, axis=1)+np.sum(calB, axis=1)

        if grad:
            dgi=-4*a*pa*X[:, :-1] +2*pb
            dB=-c*d*np.cos(d*(1-X))
            #
            dp=np.zeros_like(X)
            dp[:, 0]=dgi[:, 0]+dB[:, 0]
            dp[:, 1:-1]=dgi[:, 1:] +2*a*pa[:, :-1] +dB[:, 1:-1]
            dp[:, -1]=2*a*pa[:, -1]+dB[:, -1]

        return (p, dp) if grad else p


funRosenbrockMS = FunRosenbrockMS()
