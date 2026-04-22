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


class FunChichinadze(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 04/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 global minimum
        f(x1,x2)=-42.94438701899098  for (x1,x2)=(6.189866586965680,0.5)
        Design space: -500<x1,x2<500
        """

        #constants
        a=12
        b=11
        c=10
        d=np.pi/2
        e=8
        f=5*d
        g=0.2*np.sqrt(5)
        h=1/2

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluations and derivatives
        pa=xxx**2-a*xxx+b
        pb=c*np.cos(d*xxx)+e*np.sin(f*xxx)
        pc=np.exp(-h*(yyy-h)**2)
        #
        p=pa+pb+g*pc

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*xxx-a -c*d*np.sin(d*xxx)+e*f*np.cos(f*xxx)
            dp[:, 1]=-2*h*g*yyy*pc

        return (p, dp) if grad else p


funChichinadze = FunChichinadze()
