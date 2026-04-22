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


class FunRotatedEllipse2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Rotated Ellipse function 2
        L. LAURENT -- 14/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,0)=0
        Design space -500<xi<500
        """

        #constants
        a=7
        b=6*np.sqrt(3)
        c=13

        #evaluation and derivatives
        p=a*X[:, 0]**2-b*X[:, 0]*X[:, 1]+c*X[:, 1]**2


        if grad:
            #
            dp=np.zeros_like(X)
            dp[:, 0]=2*a*X[:, 0]-b*X[:, 1]
            dp[:, 1]=-b*X[:, 0]+2*c*X[:, 1]

        return (p, dp) if grad else p


funRotatedEllipse2 = FunRotatedEllipse2()
