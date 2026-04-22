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


class FunSchaffer6(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schaffer 06 function
        L. LAURENT -- 27/04/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for xi=0
        Design space: -100<xi<100
        """
        #constants
        a=0.5
        b=1
        c=1e-3

        #evaluation and derivatives
        xi=X[:, :-1]
        xii=X[:, 1:]
        #
        xA=xii**2+xi**2
        #
        sqA=np.sqrt(xA)
        sA=np.sin(sqA)
        pA=sA**2-a
        #
        pB=b+c*xA**2
        #
        pt=pA/pB
        #
        p=np.sum(a+pt, axis=1)
        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            cA=np.cos(sqA)
            #
            dp[:, 0]=(2*xi[:, 0]/sqA[:, 0]*sA[:, 0]*cA[:, 0]*pB[:, 0] -4*c*xi[:, 0]*xA[:, 0]*pA[:, 0])/pB[:, 0]**2
            #
            dp[:, -1]=(2*xii[:, -1]/sqA[:, -1]*sA[:, -1]*cA[:, -1]*pB[:, -1] -4*c*xii[:, -1]*xA[:, -1]*pA[:, -1])/pB[:, -1]**2
            #
            dp[:, 1:-1]=(2*xi[:, 1:]/sqA[:, 1:]*sA[:, 1:]*cA[:, 1:]*pB[:, 1:] -4*c*xi[:, 1:]*xA[:, 1:]*pA[:, 1:])/pB[:, 1:]**2 +(2*xii[:, :-1]/sqA[:, :-1]*sA[:, :-1]*cA[:, :-1]*pB[:, :-1] -4*c*xii[:, :-1]*xA[:, :-1]*pA[:, :-1])/pB[:, :-1]**2

        return (p, dp) if grad else p


funSchaffer6 = FunSchaffer6()
