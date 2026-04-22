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


class FunRosenbrockM(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Modified Rosenbrock function
        L. LAURENT -- 14/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(-1,-1)=0
        Design space -2<xi<2
        """

        #constants
        a=74
        b=100
        c=1
        d=400
        e=0.1

        #evaluation and derivatives
        pa=(X+c)**2/e
        calA=d*np.exp(-np.sum(pa, axis=1))

        #
        p=a+b*(X[:, 1]-X[:, 0]**2)**2+(c-X[:, 0])**2-calA

        if grad:
            #
            dp=np.zeros_like(X)
            dp[:, 0]=-4*b*X[:, 0]*(X[:, 1]-X[:, 0]**2) -2*(c-X[:, 0])+ 2/e*(X[:, 0]+c)*calA
            dp[:, 1]=2*b*(X[:, 1]-X[:, 0]**2) +2/e*(X[:, 1]+c)*calA

        return (p, dp) if grad else p


funRosenbrockM = FunRosenbrockM()
