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


class FunRosenbrock(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 12/05/2010 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(1,...,1)=0
        Design space -30<xi<30
        """

        #constants
        a=100
        b=1

        #evaluation and derivatives
        pa=X[:, 1:]-X[:, :-1]**2
        pb=X[:, :-1]-b
        cal=a*pa**2+pb**2
        #
        p=np.sum(cal, axis=1)

        if grad:
            dgi=-4*a*pa*X[:, :-1] +2*pb
            #
            dp=np.zeros_like(X)
            dp[:, 0]=dgi[:, 0]
            dp[:, 1:-1]=dgi[:, 1:]+2*a*pa[:, :-1]
            dp[:, -1]=2*a*pa[:, -1]

        return (p, dp) if grad else p


funRosenbrock = FunRosenbrock()
