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


class FunUrsem3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Ursem 3 function
        L. LAURENT -- 24/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,0)=-4.8168
        Design space -2<x1<2 -1.5<x<1.5
        """

        #constants
        a=2.2*np.pi
        b=np.pi/2
        c=2
        d=3

        #evaluation and derivatives
        pa=-np.sin(a*X+b)
        pb=(c-np.abs(X))/c
        pc=(d-np.abs(X))/c
        #
        pt=pa*pb*pc
        #
        p=np.sum(pt, axis=1)
        #
        if grad:
            #
            dpa=-a*np.cos(a*X+b)
            dpb=-np.sign(X)/c
            dpc=-np.sign(X)/c
            #
            dp=dpa*pb*pc+pa*dpb*pc+pa*pb*dpc

        return (p, dp) if grad else p


funUrsem3 = FunUrsem3()
