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


class FunWayburnSeader2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Wayburn-Seader 2 function
        L. LAURENT -- 27/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0.2,1)=0
        Design space -500<xi<500
        """

        #constants
        a=1.613
        b=4
        c=0.3125
        d=1.625
        e=1

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        pa=a-b*(x-c)**2-b*(y-d)**2
        pb=y-e
        #
        p=pa**2+pb**2

        if grad:
            dp=np.zeros_like(X)
            dp[:, 0]=-4*b*(x-c)*pa
            dp[:, 1]=-4*b*(y-d)*pa+2*pb

        return (p, dp) if grad else p


funWayburnSeader2 = FunWayburnSeader2()
