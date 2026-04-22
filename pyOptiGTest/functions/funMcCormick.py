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


class FunMcCormick(TestFunction):
    def evaluate(self, X, grad=False):
        """
        McCormick's function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=-1.9133 for (x1,x2)=(?0.547,?1.547)
        Design space: -1.5<x1<4 & -3<x2<3
        """
        #constants
        a=3/2
        b=5/2
        c=1

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=xxx-yyy
        pb=xxx+yyy
        p=np.sin(pb)+pa**2-a*xxx+b*yyy+c

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=np.cos(pb)+2*pa-a
            dp[:, 1]=np.cos(pb)-2*pa+b

        return (p, dp) if grad else p


funMcCormick = FunMcCormick()
