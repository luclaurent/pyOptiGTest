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


class FunLevy03(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Levy 03 function
        L. LAURENT -- 17/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for xi=1
        Design space: -10<xi<10
        """
        #constants
        a=np.pi
        b=1
        c=4
        d=10

        #evaluation and derivatives
        y=b+(X-b)/c
        #
        pa=np.sin(a*y[:, 0])
        pb=y[:, :-1]-b
        sc=np.sin(a*y[:, :-1]+b)
        pc=b+d*sc**2
        pd=y[:, -1]-b
        se=np.sin(2*a*y[:, -1])
        pe=b+se**2
        #
        p=pa**2+np.sum(pb**2*pc, axis=1)+pd**2*pe

        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            cc=np.cos(a*y[:, :-1]+b)
            ce=np.cos(2*a*y[:, -1])
            #
            dp[:, 0]=2*a/c*pa*np.cos(a*y[:, 0]) +2/c*pb[:, 0]*pc[:, 0] +2*d*a/c*sc[:, 0]*cc[:, 0]*pb[:, 0]**2
            dp[:, -1]=2/c*pd*pe +4*a/c*se*ce*pd**2
            dp[:, 1:-1]=2/c*pb[:, 1:]*pc[:, 1:] +2*d*a/c*sc[:, 1:]*cc[:, 1:]*pb[:, 1:]**2

        return (p, dp) if grad else p


funLevy03 = FunLevy03()
