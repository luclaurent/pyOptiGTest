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


class FunPathological(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Pathological function
        L. LAURENT -- 21/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-1 for xi=0
        Design space: -100<xi<100
        """
        #constants
        a=100
        b=1/2
        c=1e-3

        #evaluation and derivatives
        xi=X[:, :-1]
        xii=X[:, 1:]
        #
        xA=a*xii**2+xi**2
        xB=xi-xii
        #
        sqA=np.sqrt(xA)
        sA=np.sin(sqA)
        pA=sA**2-b
        #
        pB=c*xB**4+b
        #
        pt=pA/pB
        #
        p=np.sum(pt, axis=1)
        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            cA=np.cos(sqA)
            #
            dp[:, 0]=(2*X[:, 0]/sqA[:, 0]*sA[:, 0]*cA[:, 0]*pB[:, 0] -4*c*xB[:, 0]**3*pA[:, 0])/pB[:, 0]**2
            #
            dp[:, -1]=(2*a*X[:, -1]/sqA[:, -1]*sA[:, -1]*cA[:, -1]*pB[:, -1] +4*c*xB[:, -1]**3*pA[:, -1])/pB[:, -1]**2
            #
            dp[:, 1:-1]=(2*X[:, 1:-1]/sqA[:, 1:]*sA[:, 1:]*cA[:, 1:]*pB[:, 1:] -4*c*xB[:, 1:]**3*pA[:, 1:])/pB[:, 1:]**2 +(2*a*X[:, 1:-1]/sqA[:, :-1]*sA[:, :-1]*cA[:, :-1]*pB[:, :-1] +4*c*xB[:, :-1]**3*pA[:, :-1])/pB[:, :-1]**2

        return (p, dp) if grad else p


funPathological = FunPathological()
