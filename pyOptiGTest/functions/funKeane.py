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


class FunKeane(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Keane's function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=-0.673668 for (x1,x2)={(0,1.39325),(1.39325,0)}
        Design space: 0<xi<10
        """

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        sa=np.sin(xxx-yyy)
        sb=np.sin(xxx+yyy)
        pa=np.sqrt(xxx**2+yyy**2)
        #
        p=sa**2*sb**2/pa

        if grad:
            ca=np.cos(xxx-yyy)
            cb=np.cos(xxx+yyy)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=(2*sa*ca*sb**2+2*sb*cb*sa**2)/pa-xxx*p/pa**2
            dp[:, 1]=(-2*sa*ca*sb**2+2*sb*cb*sa**2)/pa-yyy*p/pa**2

        return (p, dp) if grad else p


funKeane = FunKeane()
