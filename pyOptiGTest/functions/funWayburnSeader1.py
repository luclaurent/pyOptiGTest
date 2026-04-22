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


class FunWayburnSeader1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Wayburn-Seader 1 function
        L. LAURENT -- 27/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(1,2)=0
        Design space -pi<xi<pi
        """

        #constants
        a=17
        b=2
        c=4

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        pa=x**6+y**4-a
        pb=b*x+y-c
        #
        p=pa**2+pb**2

        if grad:
            dp=np.zeros_like(X)
            dp[:, 0]=12*x**5*pa+2*b*pb
            dp[:, 1]=8*y**3*pa+2*pb


        return (p, dp) if grad else p


funWayburnSeader1 = FunWayburnSeader1()
