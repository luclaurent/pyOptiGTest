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


class FunMishra06(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 6 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-2.67507597 for x=(2.94777, 1.82174)
        Design space: -10<xi<10
        """

        #constants
        a=1e-2
        b=1

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        g=np.cos(xxx)+np.cos(yyy)
        h=np.sin(xxx)+np.sin(yyy)
        kx=xxx-b
        ky=yyy-b
        pa=np.sin(g**2)**2-np.cos(h**2)**2+xxx
        #
        p=-np.log(pa**2)+a*(kx**2+ky**2)

        if grad:
            dgX=-np.sin(xxx)
            dgY=-np.sin(yyy)
            dhX=np.cos(xxx)
            dhY=np.cos(yyy)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=-2*pa*(4*dgX*g*np.cos(g**2)*np.sin(g**2) +4*dhX*h*np.sin(h**2)*np.cos(h**2)+1)/pa**2 +2*a*kx
            dp[:, 1]=-2*pa*(4*dgY*g*np.cos(g**2)*np.sin(g**2) +4*dhY*h*np.sin(h**2)*np.cos(h**2))/pa**2 +2*a*ky

        return (p, dp) if grad else p


funMishra06 = FunMishra06()
