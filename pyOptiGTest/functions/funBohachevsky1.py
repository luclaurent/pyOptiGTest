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


class FunBohachevsky1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/09/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum: f(x1,x2)=0 pour (x1,x2)=(0,0)
        Domaine d'etude de la fonction: -100<x1<100, -100<x2<100 (-2<xi<2)
        """

        # responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        p = xxx**2+2*yyy**2-0.3*np.cos(3*np.pi*xxx)-0.4*np.cos(4*np.pi*yyy)+0.7

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*xxx+0.9*np.pi*np.sin(3*np.pi*xxx)
            dp[:, 1]=4*yyy+1.6*np.pi*np.sin(4*np.pi*yyy)

        return (p, dp) if grad else p


funBohachevsky1 = FunBohachevsky1()
