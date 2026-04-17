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


class FunMieleCantrell(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Miele Cantrell's function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2,x3,x4)=0 for (x1,x2,x3,x4)=(0,1,1,1)
        Design space: -1<xi<1
        """
        #constants
        a=100

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]
        zzz=X[:, 2]
        uuu=X[:, 3]

        #evaluation and derivatives
        pa=np.exp(-xxx)-yyy
        pb=yyy-zzz
        pc=zzz-uuu
        #
        p=pa**4+a*pb**6+np.tan(pc)**4+xxx**8

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-4*np.exp(-xxx)*pa**3+8*xxx**7
            dp[:, 1]=-4*pa**3+6*a*pb**5
            dp[:, 2]=-6*a*pb**5+4*(1+np.tan(pc)**2)*np.tan(pc)**3
            dp[:, 3]=-4*(1+np.tan(pc)**2)*np.tan(pc)**3

        return (p, dp) if grad else p


funMieleCantrell = FunMieleCantrell()
