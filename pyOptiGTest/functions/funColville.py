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


class FunColville(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Colville function
        L. LAURENT -- 16/09/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2,x3,x4)=0 for (x1,x2,x3,x4)=(1,1,1,1)
        Design space: -10<xi<10
        """

        #constants
        a=100
        b=1
        c=90
        d=10.1
        e=19.8

        #evaluation and derivatives
        pa=X[:, 0]-X[:, 1]**2
        pb=b-X[:, 0]
        pc=X[:, 3]-X[:, 2]**2
        pd=b-X[:, 2]
        pe=X[:, 1]-b
        pf=X[:, 3]-b
        #
        p=a*pa**2+pb**2+c*pc**2+pd**2+d*pe**2+d*pf**2+e*pe*pf
        #
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*pa-2*pb
            dp[:, 1]=-4*a*X[:, 1]*pa+2*d*pe+e*pf
            dp[:, 2]=-4*c*X[:, 2]*pc-2*pd
            dp[:, 3]=2*c*pc+2*d*pf+e*pe

        return (p, dp) if grad else p


funColville = FunColville()
