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


class FunMishra05(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 5 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-0.119829 for x=(-1.98682, -10.0)
        Design space: -10<xi<10
        """

        #constants
        a=1e-2

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        g=np.cos(xxx)+np.cos(yyy)
        h=np.sin(xxx)+np.sin(yyy)
        pa=np.sin(g**2)**2+np.cos(h**2)**2+xxx
        #
        p=pa**2+a*(xxx+yyy)

        if grad:
            dgX=-np.sin(xxx)
            dgY=-np.sin(yyy)
            dhX=np.cos(xxx)
            dhY=np.cos(yyy)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=2*pa*(4*dgX*g*np.cos(g**2)*np.sin(g**2) -4*dhX*h*np.sin(h**2)*np.cos(h**2)+1) +a
            dp[:, 1]=2*pa*(4*dgY*g*np.cos(g**2)*np.sin(g**2) -4*dhY*h*np.sin(h**2)*np.cos(h**2)) +a

        return (p, dp) if grad else p


funMishra05 = FunMishra05()
