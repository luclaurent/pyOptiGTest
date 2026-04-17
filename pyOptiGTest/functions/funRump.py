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


class FunRump(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Rump function
        L. LAURENT -- 14/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,0)=0
        Design space -500<xi<500
        """

        #constants
        a=333.75
        b=11
        c=121
        d=2
        e=5.5
        f=2

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        p=(a-x**2)*y**6 +x**2*(b*x**2*y**2-c*y**4-d) +e*y**8+x/(f*y)

        if grad:
            #
            dp=np.zeros_like(X)
            dp[:, 0]=-2*x*y**6 +2*x*(b*x**2*y**2-c*y**4-d) +2*b*x**3*y**2 +1/(f*y)
            dp[:, 1]=6*y**5*(a-x**2) +x**2*(2*b*x**2*y-4*c*y**3) +8*e*y**7 -x/(f*y**2)

        return (p, dp) if grad else p


funRump = FunRump()
